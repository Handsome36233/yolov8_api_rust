#![allow(clippy::manual_retain)]

use std::path::PathBuf;
use std::sync::Arc;
use image::{imageops::FilterType, GenericImageView};
use ndarray::{s, Array, ArrayBase, Axis, OwnedRepr, Dim};
use ort::{inputs, CUDAExecutionProvider, Session, SessionOutputs};
use actix_web::{web, App, HttpServer, Responder, HttpResponse};
use serde::{Deserialize, Serialize};
use env_logger::Env;
use clap::{Arg, Command};

#[derive(Debug, Clone, Copy, Serialize)]
struct BoundingBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

#[derive(Deserialize)]
struct ImagePath {
    path: String,
}

fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1)) - intersection(box1, box2)
}

fn get_input_image(image_path: &PathBuf, input_width: u32, input_height: u32) -> 
	(u32, u32, ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>) {
	let original_img = image::open(image_path).unwrap();
	let (img_width, img_height) = (original_img.width(), original_img.height());
	let img = original_img.resize_exact(input_width, input_height, FilterType::CatmullRom);
	let mut input = Array::zeros((1, 3, input_height as usize, input_width as usize));
	for pixel in img.pixels() {
		let x = pixel.0 as _;
		let y = pixel.1 as _;
		let [r, g, b, _] = pixel.2.0;
		input[[0, 0, y, x]] = (r as f32) / 255.;
		input[[0, 1, y, x]] = (g as f32) / 255.;
		input[[0, 2, y, x]] = (b as f32) / 255.;
	}
	(img_width, img_height, input)
}

fn filter_boxes(outputs: SessionOutputs, input_width: u32, input_height: u32, img_width: u32, 
	img_height: u32, conf_threshold: f32) -> ort::Result<Vec<(BoundingBox, f32)>> {
	let mut boxes = Vec::new();
	let output = outputs["output0"].try_extract_tensor::<f32>()?.t().into_owned();
	let output = output.slice(s![.., .., 0]);
	for row in output.axis_iter(Axis(0)) {
		let row: Vec<_> = row.iter().copied().collect();
		let (_class_id, prob) = row
			.iter()
			// skip bounding box coordinates
			.skip(4)
			.enumerate()
			.map(|(index, value)| (index, *value))
			.reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
			.unwrap();
		if prob < conf_threshold {
			continue;
		}
		let xc = row[0] / input_width as f32 * (img_width as f32);
		let yc = row[1] / input_height as f32 * (img_height as f32);
		let w = row[2] / input_width as f32 * (img_width as f32);
		let h = row[3] / input_height as f32 * (img_height as f32);
		boxes.push((
			BoundingBox {
				x1: xc - w / 2.,
				y1: yc - h / 2.,
				x2: xc + w / 2.,
				y2: yc + h / 2.
			},
			prob
		));
	}

	boxes.sort_by(|box1, box2| box2.1.total_cmp(&box1.1));
	Ok(boxes)
}


fn process_boxes(outputs: SessionOutputs, input_width: u32, input_height: u32, img_width: u32, img_height: u32, 
	conf_threshold: f32, iou_threshold: f32) -> ort::Result<Vec<(BoundingBox, f32)>> {
    let mut boxes = filter_boxes(outputs, input_width, input_height, img_width, img_height, conf_threshold)?;
    let mut result = Vec::new();
    // nms
    while !boxes.is_empty() {
        result.push(boxes[0]);
        boxes = boxes
            .iter()
            .filter(|box1| intersection(&boxes[0].0, &box1.0) / union(&boxes[0].0, &box1.0) < iou_threshold)
            .copied()
            .collect();
    }
    Ok(result)
}


fn run_model(model: &Session, input: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>) -> ort::Result<SessionOutputs> {
	let outputs: SessionOutputs = model.run(inputs!["images" => input.view()]?)?;
	Ok(outputs)
}

async fn detect_bboxes(image_info: web::Json<ImagePath>,  model: web::Data<Arc<Session>>) -> impl Responder {
    let image_path = PathBuf::from(&image_info.path);
    // 设置置信度阈值和 IoU 阈值
    let conf_threshold: f32 = 0.3;
    let iou_threshold: f32 = 0.5;
	// 获取模型输入尺寸
	let input_shape = 	model.inputs[0].input_type.tensor_dimensions();
	println!("Input shape: {:?}", input_shape);
	let input_shape = input_shape.unwrap();
	let input_width = input_shape[3] as u32;
	let input_height = input_shape[2] as u32;

    let (img_width, img_height, input) = get_input_image(&image_path, input_width, input_height);
	let outputs: SessionOutputs = match run_model(&model, input) {
		Ok(outputs) => outputs,
		Err(e) => {
			eprintln!("Error running model: {:?}", e);
			return HttpResponse::InternalServerError().body("Error running model");
		}
	};	
    match process_boxes(outputs, input_width, input_height, img_width, img_height, conf_threshold, iou_threshold) {
        Ok(boxes) => {
            let response = serde_json::to_string(&boxes).unwrap();
			eprintln!("Response: {}", response);
            return HttpResponse::Ok().body(response)
        },
        Err(e) => {
            eprintln!("Error processing boxes: {:?}", e);
            return HttpResponse::InternalServerError().body("Error processing boxes")
        },
    }
}

#[actix_web::main]
async fn main_api(model: Arc<Session>, ip: &str, port: u16) -> Result<(), std::io::Error> {
    let bind_address = format!("{}:{}", ip, port);
    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(model.clone()))
            .route("/detect", web::post().to(detect_bboxes))
    })
    .bind(bind_address)?
    .run()
    .await?;
    Ok(())
}

fn load_model(model_path: &PathBuf) -> ort::Result<Session> {
    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;
    let model = Session::builder()?.commit_from_file(model_path)?;
    Ok(model)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init_from_env(Env::default().default_filter_or("debug"));
    let matches = Command::new("Your App")
        .version("1.0")
        .about("Object detection service")
        .arg(Arg::new("model_path")
            .help("Path to the model file")
            .required(true)
            .index(1)) 
        .arg(Arg::new("ip")
            .long("ip")
            .help("IP address to bind to")
            .default_value("127.0.0.1"))
        .arg(Arg::new("port")
            .long("port")
            .help("Port to bind to")
            .default_value("8080"))
        .get_matches();

    let model_path = PathBuf::from(matches.get_one::<String>("model_path").unwrap());
    let ip = matches.get_one::<String>("ip").unwrap();
    let port: u16 = matches.get_one::<String>("port").unwrap().parse()?;

    println!("Model path: {}", model_path.display());
    println!("Running server on {}:{}", ip, port);

    let model = load_model(&model_path)?;
    let model = Arc::new(model);
    
    main_api(model, ip, port).unwrap();

    Ok(())
}
mod model;
mod training;
mod utils;
mod data;

use crate::utils::input::get_input;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Path configuration
    let csv_path = "airquality.csv";
    let plot_path = "training_plot.png";
    let model_path = "trained_model.bin";
    let epochs = 1000;

    // Load or train model
    let trained_model = if std::path::Path::new(model_path).exists() {
        println!("Loading existing model...");
        utils::io::load_model(model_path)?
    } else {
        println!("Training new model...");
        let model = training::trainer::train_model(csv_path, epochs, plot_path)?;
        utils::io::save_model(model_path, &model)?;
        model
    };

    println!("Model ready for predictions");

    // Get user input for prediction
    println!("\nEnter air quality parameters to predict:");
    
    let pm10 = get_input("PM10: ");
    let so2 = get_input("SO2: ");
    let co = get_input("CO: ");
    let o3 = get_input("O3: ");
    let no2 = get_input("NO2: ");

    // Make prediction
    let prediction = trained_model.network.predict(
        &[pm10, so2, co, o3, no2],
        &trained_model.stats
    );

    // Display results
    prediction.display();

    Ok(())
}
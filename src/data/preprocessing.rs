use ndarray::{Array1, Array2, Axis};
use csv::Reader;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::error::Error;

#[derive(Serialize, Deserialize)]
pub struct DataStats {
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
}

pub fn load_and_preprocess_data(csv_path: &str) -> Result<(Array2<f64>, Array2<f64>, DataStats), Box<dyn Error>> {
    // Read dataset
    let file = File::open(csv_path)?;
    let mut rdr = Reader::from_reader(file);
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    for result in rdr.records() {
        let record = result?;

        // Parse data
        let pm10: f64 = record[0].parse().unwrap_or(0.0);
        let so2: f64 = record[1].parse().unwrap_or(0.0);
        let co: f64 = record[2].parse().unwrap_or(0.0);
        let o3: f64 = record[3].parse().unwrap_or(0.0);
        let no2: f64 = record[4].parse().unwrap_or(0.0);

        let category = &record[5];
        inputs.push(vec![pm10, so2, co, o3, no2]);
        outputs.push(match category {
            "BAIK" => vec![1.0, 0.0, 0.0],
            "SEDANG" => vec![0.0, 1.0, 0.0],
            "TIDAK SEHAT" => vec![0.0, 0.0, 1.0],
            _ => vec![0.0, 0.0, 0.0]
        });
    }

    // Convert to ndarray
    let x = Array2::from_shape_vec((inputs.len(), 5), inputs.concat())?;
    let y = Array2::from_shape_vec((outputs.len(), 3), outputs.concat())?;

    // Normalize input
    let mean = x.mean_axis(Axis(0)).unwrap();
    let std = x.std_axis(Axis(0), 1.0);
    let x_normalized = (x - &mean) / &std;

    Ok((x_normalized, y, DataStats { mean, std }))
}
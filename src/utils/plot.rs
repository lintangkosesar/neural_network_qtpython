use crate::training::history::TrainingHistory;
use plotters::prelude::*;
use std::error::Error;

pub fn create_plot(history: &TrainingHistory, path: &str) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_epoch = *history.epochs.last().unwrap_or(&1) as u32;
    let max_loss = history.losses.iter().cloned().fold(f64::NAN, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Progress", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0u32..max_epoch, 0f64..1f64)?;

    chart.configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_desc("Epoch")
        .y_desc("Value")
        .draw()?;

    // Plot accuracy
    chart.draw_series(LineSeries::new(
        history.epochs.iter().zip(history.accuracies.iter()).map(|(&x, &y)| (x as u32, y)),
        &RED,
    ))?.label("Accuracy")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Plot scaled loss
    let loss_scale = 1.0 / max_loss;
    chart.draw_series(LineSeries::new(
        history.epochs.iter().zip(history.losses.iter()).map(|(&x, &y)| (x as u32, y * loss_scale)),
        &BLUE,
    ))?.label("Loss (scaled)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}
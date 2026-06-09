use criterion::{criterion_group, criterion_main, Criterion};
use microflow::buffer::Buffer2D;
use microflow_macros::model;

#[model("models/nano_u.tflite")]
struct NanoU;

fn nano_u_model(c: &mut Criterion) {
    let input = [Buffer2D::from_element([0.0f32, 0.0, 0.0])];

    c.bench_function("nano_u_model", |b| b.iter(|| NanoU::predict(input)));
}

criterion_group!(benches, nano_u_model);
criterion_main!(benches);

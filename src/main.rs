#![allow(non_snake_case)]

use gradrs::Value;

fn main() -> anyhow::Result<()> {
    lol();

    Ok(())
}

fn lol() {
    // inputs
    let x1 = Value::from(2.0).with_label("x1");
    let x2 = Value::from(0.0).with_label("x2");
    // weights
    let w1 = Value::from(-3.0).with_label("w1");
    let w2 = Value::from(1.0).with_label("w2");
    // bias
    let b = Value::from(6.8813735870195432).with_label("b");
    let x1w1 = (x1 * w1).with_label("x1w1");
    let x2w2 = (x2 * w2).with_label("x2w2");
    let x1w1x2w2 = (x1w1 + x2w2).with_label("x1w1 + x2w2");
    let n = (x1w1x2w2 + b).with_label("n");
    let o = n.tanh().with_label("o");

    o.backward();
    println!("{:?}", o);
}

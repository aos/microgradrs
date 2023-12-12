#![allow(non_snake_case)]

use gradrs::layer::Layer;
use gradrs::mlp::MLP;
use gradrs::value::Value;

fn main() -> anyhow::Result<()> {
    c4();

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
    let mut o = n.tanh().with_label("o");

    o.backward();
    println!("{:?}", o);
}

fn lol2() {
    let a = Value::from(2.0).with_label("a");
    let b = Value::from(4.0).with_label("b");
    let mut c = a / b;

    c.backward();
    println!("{:?}", c);
}

fn c3() {
    let z = Layer::new(2, 3);
    let x = z.call(vec![2.0.into(), 3.0.into()]);
    for i in x {
        println!("{}", i);
    }
}

fn c4() {
    let x = vec![2.0, 3.0, -1.0];
    let z = MLP::new(3, vec![4, 4, 1]);
    let n = z.call(x.iter().map(|n| Value::from(*n)).collect::<Vec<_>>());

    for i in n {
        println!("{}", i);
    }
}

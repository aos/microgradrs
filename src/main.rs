#![allow(non_snake_case)]

use gradrs::Value;

fn main() -> anyhow::Result<()> {
    let L = lol();

    println!("{}", L);

    Ok(())
}

fn lol() -> f64 {
    let h = 0.001;

    let a = Value::from(2.0).with_label("a");
    let b = Value::from(-3.0).with_label("b");
    let c = Value::from(10.0).with_label("c");
    let e = (a * b).with_label("e");

    let d = (e + c).with_label("d");
    let f = Value::from(-2.0).with_label("f");
    let L1 = (d * f).with_label("L1");

    let a = Value::from(2.0).with_label("a");
    let b = Value::from(-3.0).with_label("b");
    let c = Value::from(10.0 + h).with_label("c");
    let e = (a * b).with_label("e");

    let d = (e + c).with_label("d");
    let f = Value::from(-2.0).with_label("f");
    let L2 = (d * f).with_label("L1");

    (L2.data() - L1.data()) / h
}

fn l2() {
    let a = Value::from(2.0).with_label("a");
    let d = &a + &a;
}

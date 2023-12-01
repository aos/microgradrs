#![allow(non_snake_case)]

use gradrs::Value;

fn main() -> anyhow::Result<()> {
    let a = Value::new(2.0, Some("a"));
    let b = Value::new(-3.0, Some("b"));
    let c = Value::new(10.0, Some("c"));
    let mut e = a * b;
    e.label("e");
    let mut d = e + c;
    d.label("d");
    let f = Value::new(-2.0, Some("f"));
    let mut L = d * f;
    L.label("L");

    println!("{}", L);

    Ok(())
}

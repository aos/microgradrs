use gradrs::{Value, MLP};

fn main() {
    let z = MLP::new(3, vec![4, 4, 1]);

    // tiny data set
    let xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ];
    // desired targets
    let ys = [1.0, -1.0, -1.0, 1.0];

    let o = xs
        .iter()
        .map(|x| z.call(x.iter().map(|n| Value::from(*n)).collect::<Vec<_>>()))
        .collect::<Vec<_>>();

    for i in o {
        println!("{}", i[0]);
    }
}

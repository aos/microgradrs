use std::fmt;
use std::ops::{Add, Deref, Mul};
use std::rc::Rc;
use std::cell::{Ref, RefCell};

#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueInternal>>);

impl Value {
    pub fn from<T>(t: T) -> Self
    where T: Into<Value>,
    {
        t.into()
    }

    fn new(t: ValueInternal) -> Self {
        Self(Rc::new(RefCell::new(t)))
    }

    pub fn with_label(self, label: &str) -> Value {
        self.0.borrow_mut().label = Some(label.to_string());
        self
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }
}

impl<T: Into<f64>> From<T> for Value {
    fn from(t: T) -> Value {
        Value::new(ValueInternal::new(t.into(), None, None, Vec::new(), None))
    }
}

impl Add<Value> for Value {
    type Output = Value;

    fn add(self, other: Value) -> Self::Output {
        add(&self, &other)
    }
}

impl<'a, 'b> Add<&'b Value> for &'a Value {
    type Output = Value;

    fn add(self, other: &'b Value) -> Self::Output {
        add(self, other)
    }
}

fn add(a: &Value, b: &Value) -> Value {
    let result = a.data() + b.data();

    let prop_fn: PropagateFn = |value| {
        let mut first = value.previous[0].0.borrow_mut();
        let mut second = value.previous[1].0.borrow_mut();

        first.gradient += value.gradient;
        second.gradient += value.gradient;
    };

    Value::new(ValueInternal::new(
        result,
        None,
        Some("+".to_string()),
        vec![a.clone(), b.clone()],
        Some(prop_fn),
    ))
}

impl Mul<Value> for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Self::Output {
        mul(&self, &other)
    }
}

fn mul(a: &Value, b: &Value) -> Value {
    let result = a.data() * b.data();

    //let prop_fn: PropagateFn = |value| {
    //    let mut first = value.previous
    //};
    Value::new(ValueInternal::new(
        result,
        None,
        Some("*".to_string()),
        vec![a.clone(), b.clone()],
        None,
    ))
}

// impl Deref for Value {
//     type Target = Rc<RefCell<ValueInternal>>;
//
//     fn deref(&self) -> &Self::Target {
//         &self.0
//     }
// }

type PropagateFn = fn(value: &Ref<ValueInternal>);

struct ValueInternal {
    data: f64,
    gradient: f64,
    label: Option<String>,
    operation: Option<String>,
    previous: Vec<Value>,
    propagate: Option<PropagateFn>,
}

impl ValueInternal {
    fn new(
        data: f64,
        label: Option<String>,
        op: Option<String>,
        prev: Vec<Value>,
        propagate: Option<PropagateFn>,
    ) -> Self {
        ValueInternal {
            data,
            gradient: 0.0,
            label,
            operation: op,
            previous: prev,
            propagate,
        }
    }
}

#![allow(dead_code)]
use crate::matrix::ToMatrix;

pub mod matrix;
fn main() {
    let x= [[1,0,0],[0,1,0],[0,0,1]].to_matrix();
    
    // let y = &x*&x;
    println!("hello world");
}

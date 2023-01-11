#![allow(dead_code)]
use crate::matrix::ToMatrix;

pub mod matrix;
fn main() {
    let x= [[1,0,0],[0,1,0],[0,0,1]].to_matrix();
    let y = [[1,2,3],[4,5,6],[7,8,9]].to_matrix();
    let y = &x * &x * &y * 8.0;

    println!("{}",y);
    // let y = &x*&x;
    println!("hello world");
}

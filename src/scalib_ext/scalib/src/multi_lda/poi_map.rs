use itertools::Itertools;
use serde::{Deserialize, Serialize};

use super::Var;
use crate::{Result, ScalibError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoiMap {
    new2old: Vec<u32>,
    new_poi_vars: Vec<Vec<u32>>,
}

impl PoiMap {
    pub fn new(ns: u32, poi_vars: &[Vec<u32>]) -> Result<Self> {
        let mut used_pois = vec![false; ns as usize];
        for poi in poi_vars.iter().flat_map(|x| x.iter()) {
            *used_pois
                .get_mut(*poi as usize)
                .ok_or(ScalibError::PoiOutOfBound)? = true;
        }
        let new2old = used_pois
            .iter()
            .positions(|x| *x)
            .map(|x| x as u32)
            .collect_vec();
        let mut cnt: u32 = 0;
        let old2new = used_pois
            .iter()
            .map(|x| {
                x.then(|| {
                    cnt += 1;
                    cnt - 1
                })
            })
            .collect_vec();
        let new_poi_vars = poi_vars
            .iter()
            .map(|pois| pois.iter().map(|x| old2new[*x as usize].unwrap()).collect())
            .collect();

        Ok(Self {
            new2old,
            new_poi_vars,
        })
    }
    pub fn len(&self) -> usize {
        self.new2old.len()
    }
    pub fn kept_indices(&self) -> &[u32] {
        self.new2old.as_slice()
    }
    pub fn new_pois(&self, var: Var) -> &[u32] {
        &self.new_poi_vars[var as usize]
    }
    pub fn new_pois_vars(&self) -> &[Vec<u32>] {
        &self.new_poi_vars
    }
    pub fn n_pois(&self, var: Var) -> usize {
        self.new_pois(var).len()
    }
    /// POI blocks for MultiLda.predict_proba
    pub fn poi_blocks(&self) -> Vec<Vec<Vec<u16>>> {
        use super::POI_BLOCK_SIZE;
        assert!(POI_BLOCK_SIZE < (u16::MAX as usize));
        let n_poi_blocks = self.len().div_ceil(POI_BLOCK_SIZE);
        self.new_pois_vars()
            .iter()
            .map(|pois| {
                let mut res = vec![vec![]; n_poi_blocks];
                for poi in pois.iter() {
                    let poi = *poi as usize;
                    res[poi / POI_BLOCK_SIZE].push((poi % POI_BLOCK_SIZE) as u16);
                }
                res
            })
            .collect()
    }
}

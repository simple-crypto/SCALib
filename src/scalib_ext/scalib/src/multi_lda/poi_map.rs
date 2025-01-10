use itertools::Itertools;
use serde::{Deserialize, Serialize};

use super::Var;
use crate::{Result, ScalibError};

use std::borrow::Borrow;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoiMap {
    new2old: Vec<u32>,
    new_poi_vars: Vec<Vec<u32>>,
}

impl PoiMap {
    pub fn new<I: IntoIterator<Item = impl Borrow<u32>>>(
        ns: usize,
        poi_vars: impl IntoIterator<Item = I> + Clone,
    ) -> Result<Self> {
        assert!(poi_vars
            .clone()
            .into_iter()
            .all(|p| p.into_iter().map(|x| *x.borrow()).is_sorted()));
        let mut used_pois = vec![false; ns as usize];
        for poi in poi_vars.clone().into_iter().flat_map(I::into_iter) {
            *used_pois
                .get_mut(*poi.borrow() as usize)
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
            .into_iter()
            .map(|pois| {
                pois.into_iter()
                    .map(|x| old2new[*x.borrow() as usize].unwrap())
                    .collect()
            })
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
    pub fn select_vars(&self, vars: &[super::Var]) -> Result<Self> {
        let sub_map = Self::new(self.len(), vars.iter().map(|v| self.new_pois(*v)))?;
        let full_map = Self {
            new2old: sub_map
                .new2old
                .iter()
                .map(|x| self.new2old[*x as usize])
                .collect(),
            new_poi_vars: sub_map.new_poi_vars.clone(),
        };
        Ok(full_map)
    }
    pub fn mapped_pairs(&self, var: Var) -> impl Iterator<Item = (u32, u32)> + '_ {
        let pois = self.new_pois(var);
        super::MultiLdaAccConf::pairs_n(pois.len() as u32)
            .map(|(i, j)| (pois[i as usize], pois[j as usize]))
    }
}

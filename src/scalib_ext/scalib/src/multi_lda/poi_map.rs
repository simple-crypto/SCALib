use serde::{Deserialize, Serialize};

use super::Var;
use crate::{Result, ScalibError};

use std::borrow::Borrow;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoiMap {
    new2old: Vec<u32>,
    new_poi_vars: Vec<Vec<u32>>,
}

//#[test]
//fn test_dev()  {
//    let test_a = vec![(1,2),(3,4),(5,6)];
//    let testoption:Vec<Option<u16>> = vec![None;10];
//    assert!(false, "Test coucou {:?}", testoption);
//}

impl PoiMap {
    pub fn new<I: IntoIterator<Item = impl Borrow<u32>>>(
        ns: usize,
        poi_vars: impl IntoIterator<Item = I> + Clone,
    ) -> Result<Self> {
        assert!(poi_vars
            .clone()
            .into_iter()
            .all(|p| p.into_iter().map(|x| *x.borrow()).is_sorted()));

        let mut new2old = Vec::new();
        let mut old2new: Vec<Option<u32>> = vec![None; ns as usize];
        let mut new_poi_vars = Vec::new();
        for pois in poi_vars {
            let new_poi_var = pois
                .into_iter()
                .map(|poi| -> Result<u32> {
                    let poi = *poi.borrow();
                    let new_poi = old2new
                        .get_mut(poi as usize)
                        .ok_or(ScalibError::PoiOutOfBound)?;
                    Ok(if let Some(new_poi) = new_poi {
                        *new_poi
                    } else {
                        let n_pois = new2old.len() as u32;
                        *new_poi = Some(n_pois);
                        new2old.push(poi);
                        n_pois
                    })
                })
                .collect::<Result<_>>()?;
            new_poi_vars.push(new_poi_var);
        }
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
    pub fn poi_blocks(&self) -> Vec<Vec<(std::ops::Range<usize>, Vec<u16>)>> {
        use super::POI_BLOCK_SIZE;
        assert!(POI_BLOCK_SIZE < (u16::MAX as usize));
        let n_poi_blocks = self.len().div_ceil(POI_BLOCK_SIZE);
        self.new_pois_vars()
            .iter()
            .map(|pois| {
                let mut res = vec![(0..0, vec![]); n_poi_blocks];
                for poi in pois.iter() {
                    let poi = *poi as usize;
                    res[poi / POI_BLOCK_SIZE]
                        .1
                        .push((poi % POI_BLOCK_SIZE) as u16);
                }
                let mut npois = 0;
                for (r, pois) in res.iter_mut() {
                    let new_npois = npois + pois.len();
                    *r = npois..new_npois;
                    npois = new_npois;
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
        debug_assert!(vars.iter().enumerate().all(|(i, v)| itertools::equal(
            self.new_pois(*v).iter().map(|p| self.new2old[*p as usize]),
            full_map
                .new_pois(i as Var)
                .iter()
                .map(|p| full_map.new2old[*p as usize])
        )));
        Ok(full_map)
    }
    pub fn mapped_pairs(&self, var: Var) -> impl Iterator<Item = (u32, u32)> + '_ {
        let pois = self.new_pois(var);
        super::MultiLdaAccConf::pairs_n(pois.len() as u32)
            .map(|(i, j)| (pois[i as usize], pois[j as usize]))
    }
}

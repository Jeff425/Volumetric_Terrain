
#[derive(Clone)]
pub struct VolumetricTerrainDataNode {
    side_texture_index: u32,
    top_texture_index: u32,
    bottom_texture_index: u32,
}

impl VolumetricTerrainDataNode {
    pub fn new(side_texture_index: u32, top_texture_index: u32, bottom_texture_index: u32) -> Self {
        Self {
            side_texture_index,
            top_texture_index,
            bottom_texture_index,
        }
    }
}

#[derive(Clone)]
pub struct VolumetricTerrainData {
    nodes: Vec<VolumetricTerrainDataNode>,
}

impl VolumetricTerrainData {
    pub fn new(nodes: Vec<VolumetricTerrainDataNode>) -> Self {
        Self {
            nodes
        }
    }

    pub fn get_texture_array(&self, index: usize) -> [u32; 3] {
        [
            self.nodes[index].side_texture_index,
            self.nodes[index].top_texture_index,
            self.nodes[index].bottom_texture_index,
        ]
    }
}
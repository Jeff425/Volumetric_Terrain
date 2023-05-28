use std::sync::{RwLock, Arc};

use bevy_render::prelude::Mesh;
use bevy_tasks::{Task, AsyncComputeTaskPool};

use super::{volumetric_terrain_data::VolumetricTerrainData, volumetric_terrain_mesh::extract_volumetric_terrain_mesh};

pub struct VolumetricTerrainValues {
    density: Vec<Vec<Vec<i8>>>,
    terrain_index: Vec<Vec<Vec<u8>>>,
}

impl VolumetricTerrainValues {
    fn new(x_size: usize, y_size: usize, z_size: usize) -> Self {
        Self {
            density: vec![vec![vec![0i8; z_size]; y_size]; x_size],
            terrain_index: vec![vec![vec![0u8; z_size]; y_size]; x_size],
        }
    }

    pub fn get_density(&self, x: usize, y: usize, z: usize) -> i8 {
        self.density[x][y][z]
    }

    pub fn get_terrain_index(&self, x: usize, y: usize, z: usize) -> usize {
        self.terrain_index[x][y][z] as usize
    }

    pub fn get_regular_corners(&self, x: usize, y: usize, z: usize, lod_gap: usize) -> [i16; 8] {
        [
            self.density[x][y][z],
            self.density[x+lod_gap][y][z],
            self.density[x][y+lod_gap][z],
            self.density[x+lod_gap][y+lod_gap][z],
            self.density[x][y][z+lod_gap],
            self.density[x+lod_gap][y][z+lod_gap],
            self.density[x][y+lod_gap][z+lod_gap],
            self.density[x+lod_gap][y+lod_gap][z+lod_gap],
        ].map(Into::<i16>::into)
    }

    // Use the corners from get_regular_corners
    pub fn get_regular_case_code(corner: &[i16; 8]) -> i16 {
        ((corner[0] >> 15) & 0x01)
        | ((corner[1] >> 14) & 0x02)
        | ((corner[2] >> 13) & 0x04)
        | ((corner[3] >> 12) & 0x08)
        | ((corner[4] >> 11) & 0x10)
        | ((corner[5] >> 10) & 0x20)
        | ((corner[6] >> 9) & 0x40)
        | ((corner[7] >> 8) & 0x80)
    }
}

pub struct VolumetricTerrainFieldMeta {
    x_size: usize,
    y_size: usize,
    z_size: usize,
    terrain_data: VolumetricTerrainData,
}

impl VolumetricTerrainFieldMeta {
    pub fn get_sizes(&self) -> (usize, usize, usize) {
        (self.x_size, self.y_size, self.z_size)
    }

    pub fn get_texture_array(&self, index: usize) -> [u32; 3] {
        self.terrain_data.get_texture_array(index)
    }
}

enum TerrainValueUpdate {
    Density(usize, usize, usize, i8),
    TerrainIndex(usize, usize, usize, u8),
}

pub struct VolumetricTerrainField {
    data: Arc<VolumetricTerrainFieldMeta>,
    // primary values are written immediately when called on the calling thread
    // primary values should be read for quick operations, not for generating meshes
    primary_values: Arc<RwLock<VolumetricTerrainValues>>,
    // secondary values are only used when generating meshes, and updates have to be invalidated
    secondary_values: Arc<RwLock<VolumetricTerrainValues>>,
    secondary_update_vector: Vec<TerrainValueUpdate>,
}

impl VolumetricTerrainField {
    pub fn new(x_size: usize, y_size: usize, z_size: usize, terrain_data: VolumetricTerrainData) -> Self {
        Self {
            data: Arc::new(VolumetricTerrainFieldMeta {x_size, y_size, z_size, terrain_data}),
            primary_values: Arc::new(RwLock::new(VolumetricTerrainValues::new(x_size, y_size, z_size))),
            secondary_values: Arc::new(RwLock::new(VolumetricTerrainValues::new(x_size, y_size, z_size))),
            secondary_update_vector: Vec::new(),
        }
    }

    pub fn set_density(&mut self, x: usize, y: usize, z: usize, density: i8) {
        self.primary_values.write().unwrap().density[x][y][z] = density;
        self.secondary_update_vector.push(TerrainValueUpdate::Density(x, y, z, density));
    }

    pub fn set_terrain_index(&mut self, x: usize, y: usize, z: usize, terrain_index: u8) {
        self.primary_values.write().unwrap().terrain_index[x][y][z] = terrain_index;
        self.secondary_update_vector.push(TerrainValueUpdate::TerrainIndex(x, y, z, terrain_index));
    }

    // Drains the update queue and spawns a thread to update the terrain in a non-blocking way
    // Should this return a task that resolves to a list of affected chunk coordinates?
    pub fn update_thread_safe_values(&mut self) {
        let thread_pool = AsyncComputeTaskPool::get();
        let update_vector: Vec<_> = self.secondary_update_vector.drain(..).collect();
        let secondary_ref = self.secondary_values.clone();
        thread_pool.spawn(async move {
            let mut secondary_values = secondary_ref.write().unwrap();
            for updates in update_vector {
                match updates {
                    TerrainValueUpdate::Density(x, y, z, density) => secondary_values.density[x][y][z] = density,
                    TerrainValueUpdate::TerrainIndex(x, y, z, terrain_index) => secondary_values.terrain_index[x][y][z] = terrain_index,
                }
            }
        }).detach();
    }

    // Syncronously drains the update queue and updates the terrain in a blocking way
    pub fn update_thread_safe_values_sync(&mut self) {
        let update_vector: Vec<_> = self.secondary_update_vector.drain(..).collect();
        let mut secondary_values = self.secondary_values.write().unwrap();
        for updates in update_vector {
            match updates {
                TerrainValueUpdate::Density(x, y, z, density) => secondary_values.density[x][y][z] = density,
                TerrainValueUpdate::TerrainIndex(x, y, z, terrain_index) => secondary_values.terrain_index[x][y][z] = terrain_index,
            }
        }
    }

    // Spawns a thread to extract the mesh
    pub fn extract_mesh(
        &mut self, 
        x_min: usize,
        y_min: usize,
        z_min: usize,
        lod: usize,
        max_lod: usize,
    ) -> Task<Mesh> {
        let thread_pool = AsyncComputeTaskPool::get();
        let meta_ref = self.data.clone();
        let secondary_ref = self.secondary_values.clone();
        thread_pool.spawn(async move {
            extract_volumetric_terrain_mesh(&meta_ref, &secondary_ref.read().unwrap(), x_min, y_min, z_min, lod, max_lod)
        })
    }

    // Get the non-thread-safe density of the field.
    // Useful for physics
    pub fn get_density(&self, x: usize, y: usize, z: usize) -> i8 {
        self.primary_values.read().unwrap().density[x][y][z]
    }

    // Get the non-thread-safe terrain index of the field.
    // Useful for physics
    pub fn get_terrain_index(&self, x: usize, y: usize, z: usize) -> usize {
        self.primary_values.read().unwrap().terrain_index[x][y][z] as usize
    }

    // Clones the references to be used in components. Specifically physics shapes
    pub fn get_primary_arcs(&self) -> (Arc<VolumetricTerrainFieldMeta>, Arc<RwLock<VolumetricTerrainValues>>) {
        (self.data.clone(), self.primary_values.clone())
    }
}

impl Drop for VolumetricTerrainField {
    fn drop(&mut self) {
        // Wait for all threads to complete
        // Can't spawn anymore threads and Windows/MacOS fairly queues locks
        // This means once we get write access that no other threads could be running
        let _unused = self.secondary_values.write();
    }
}
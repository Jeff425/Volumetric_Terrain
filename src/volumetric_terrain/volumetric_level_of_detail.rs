use bevy_asset::{Handle, Assets};
use bevy_ecs::{prelude::{Component, Entity}, system::{Commands, Query, ResMut, Res}, query::{With, Changed, Or}};
use bevy_hierarchy::Children;
use bevy_render::prelude::Mesh;
use bevy_tasks::Task;
use futures_lite::future;
use crate::{level_of_detail::LevelOfDetail, chunk_manager::{ChunkManager, ChunkLocation, Chunk}, utils::vec3i::Vec3i};

use super::{volumetric_terrain_material::VolumetricTerrainMaterial, volumetric_terrain_mesh::TransitionSide};

enum TerrainStatus {
    Complete(Handle<Mesh>),
    Loading(Task<Mesh>),
    None,
}

#[derive(Component)]
pub struct MeshNear(u32);

impl MeshNear {
    pub fn new() -> Self {
        Self(0)
    }
}

#[derive(Component)]
pub struct TerrainLevelOfDetail {
    assets: Vec<[TerrainStatus; 2]>,
    x_min: usize,
    y_min: usize,
    z_min: usize,
    max_lod: usize,
}

impl TerrainLevelOfDetail {
    pub fn new(x_min: usize, y_min: usize, z_min: usize, max_lod: usize) -> Self {
        // Can't use vec! shortcut unless TerrainStatus implements clone/copy
        let mut assets = Vec::<[TerrainStatus; 2]>::new();
        for _i in 0..(max_lod+1) {
            assets.push([TerrainStatus::None, TerrainStatus::None]);
        }
        Self {
            assets,
            x_min,
            y_min,
            z_min,
            max_lod,
        }
    }

    // Begins creating the mesh if it hasn't already been created yet
    pub fn set_lod<F>(&mut self, mut task_return: F, next_lod: usize, chunk_location: Vec3i) where F: FnMut(Vec3i, usize, usize, usize, usize, usize) -> Task<Mesh> {
        if let TerrainStatus::None = self.assets[next_lod][0] {
            self.assets[next_lod][0] = TerrainStatus::Loading(task_return(chunk_location, self.x_min, self.y_min, self.z_min, next_lod, self.max_lod));
        }
    }

    // If wanting to add the asset manually
    pub fn insert_asset_task(&mut self, mesh: Task<Mesh>, lod: usize) {
        self.assets[lod][0] = TerrainStatus::Loading(mesh);
    }

    // Call after making changes to the field. Moves all active terrain statuses to the secondary position and wipes the primary position
    // Make sure to call set_lod afterwards to begin loading the new terrain
    pub fn invalidate(&mut self) {
        for i in 0..self.assets.len() {
            self.assets[i].rotate_right(1);
            self.assets[i][0] = TerrainStatus::None;
        }
    }
}

/*
    Updates mesh for the terrain in the given priority order. Moving to the next if not Complete:
    1. Primary Desired LOD
    2. Secondary Desired LOD
    3. Previously set mesh
*/
pub fn update_terrain_lod (
    mut chunk_manager: ResMut<ChunkManager>,
    mut commands: Commands,
    mut query: Query<(Entity, &mut TerrainLevelOfDetail, &LevelOfDetail, Option<&Handle<Mesh>>, &ChunkLocation)>,
) {
    for (entity, mut terrain_lod, level_of_detail, handle_op, chunk_location) in query.iter_mut() {
        let desired_lod = level_of_detail.desired_lod;
        let mut check_secondary = false;
        let mut set_lod = false;
        match &terrain_lod.assets[desired_lod][0] {
            TerrainStatus::Complete(handle) => {
                if handle_op.is_none() || handle_op.unwrap() != handle {
                    commands.entity(entity).insert(handle.clone());
                }
            },
            TerrainStatus::Loading(_) => {
                check_secondary = true;
            },
            TerrainStatus::None => {
                check_secondary = true;
                set_lod = true;
            },
        }
        if set_lod {
            terrain_lod.set_lod(|loc, x, y, z, lod, max_lod| chunk_manager.get_generator().spawn_terrain_lod_task(loc, x, y, z, lod, max_lod), desired_lod, chunk_location.location);
        }
        if check_secondary {
            if let TerrainStatus::Complete(handle) = &terrain_lod.assets[desired_lod][1] {
                if handle_op.is_none() || handle_op.unwrap() != handle {
                    commands.entity(entity).insert(handle.clone());
                }
            }
        } else {
            // If there was a successful setting of the terrain to any lod, then wipe ALL outdated meshes
            // This way only current meshes will be used when loading
            for i in 0..terrain_lod.assets.len() {
                terrain_lod.assets[i][1] = TerrainStatus::None;
            }
        }
    }
}

pub fn update_terrain_mesh_near (
    chunk_manager: Res<ChunkManager>,
    change_query: Query<(Entity, &ChunkLocation), (With<TerrainLevelOfDetail>, With<MeshNear>, Changed<LevelOfDetail>)>,
    lod_query: Query<(Entity, &LevelOfDetail), With<MeshNear>>,
    mut mesh_near_query: Query<&mut MeshNear>,
    chunk_parent_query: Query<&Children, With<Chunk>>,
) {
    // Loop through all changed terrains
    for (change_entity, change_location) in change_query.iter() {
        // Make sure terrain has the material
        if let Ok((_, origin_lod)) = lod_query.get(change_entity) {
            // Iterate through all sides
            for side in TransitionSide::iterator() {
                // Get the chunk neighbor on that side
                if let Some(&other_entity_chunk) = chunk_manager.cached_chunks.get(&(change_location.location + side.to_vec3i())) {
                    // Get the children of that chunk
                    if let Ok(children) = chunk_parent_query.get(other_entity_chunk) {
                        // Get the first child with the terrain material
                        for (other_entity, other_lod) in lod_query.iter_many(children) {
                            // If the origin has a larger lod, then flag the material's bit for that location
                            if origin_lod.desired_lod > other_lod.desired_lod {
                                if let Ok(mut mesh_near) = mesh_near_query.get_mut(change_entity) {
                                    mesh_near.0 |= side.to_bit();
                                }
                            // Otherwise, unset that bit
                            } else {
                                if let Ok(mut mesh_near) = mesh_near_query.get_mut(change_entity) {
                                    mesh_near.0 &= side.to_bit_inverse();
                                }
                            }
                            // Check the same for the other LOD, but flipping the side
                            if other_lod.desired_lod > origin_lod.desired_lod {
                                if let Ok(mut mesh_near) = mesh_near_query.get_mut(other_entity) {
                                    mesh_near.0 |= side.flip().to_bit();
                                }
                            }
                            else {
                                if let Ok(mut mesh_near) = mesh_near_query.get_mut(other_entity) {
                                    mesh_near.0 &= side.flip().to_bit_inverse();
                                }
                            }
                            // Only compare to one terrain per chunk
                            break;
                        }
                    }
                }
            }
        }
    }
}

pub fn sync_mesh_near_to_material (
    query: Query<(&MeshNear, &Handle<VolumetricTerrainMaterial>), Or<(Changed<MeshNear>, Changed<Handle<VolumetricTerrainMaterial>>)>>,
    mut materials: ResMut<Assets<VolumetricTerrainMaterial>>,
) {
    for (mesh_near, handle) in query.iter() {
        if let Some(mut material) = materials.get_mut(handle) {
            material.mesh_near = mesh_near.0;
        }
    }
}

// Completes any tasks and sets them to handles for use
// Completes both primary and secondary tasks
pub fn complete_terrain_meshes (
    mut query: Query<&mut TerrainLevelOfDetail>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    for mut terrain_lod in query.iter_mut() {
        for i in 0..terrain_lod.assets.len() {
            for j in 0..2 {
                let mut next_handle = Option::<Handle<Mesh>>::None;
                if let TerrainStatus::Loading(task) = &mut terrain_lod.assets[i][j] {
                    if let Some(mesh) = future::block_on(future::poll_once(task)) {
                        // Task has completed, create the handle for the mesh
                        if mesh.indices().is_some() && mesh.indices().unwrap().len() > 0 {
                            next_handle = Some(meshes.add(mesh));
                        } else {
                            // If the mesh is blank, don't create a handle for the asset
                            next_handle = Some(Handle::<Mesh>::default());
                        }
                        
                    }
                }
                if let Some(handle) = next_handle {
                    // Update status to the completed handle
                    terrain_lod.assets[i][j] = TerrainStatus::Complete(handle);
                }
            }
        }
    }
}
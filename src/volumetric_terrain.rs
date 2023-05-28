use bevy_app::Plugin;
use bevy_asset::{HandleUntyped, load_internal_asset};
use bevy_ecs::schedule::{IntoSystemConfigs, IntoSystemConfig};
use bevy_pbr::MaterialPlugin;
use bevy_reflect::TypeUuid;
use bevy_render::{mesh::MeshVertexAttribute, render_resource::Shader};
use crate::RPGCoreSet;

use self::volumetric_terrain_material::VolumetricTerrainMaterial;
use self::volumetric_level_of_detail::{update_terrain_lod, update_terrain_mesh_near, complete_terrain_meshes, sync_mesh_near_to_material};

// Volumetric terrain using the transvoxel technique described her https://transvoxel.org/
// Using the exact sizes the paper recommends
pub mod volumetric_terrain_material;
pub mod volumetric_terrain_field;
pub mod volumetric_terrain_data;
pub mod volumetric_terrain_mesh;
pub mod volumetric_level_of_detail;
mod data;

const BASE_ATTRIBUTE_ID: usize = 1000000000;
pub const ATTRIBUTE_SECONDARY_POSITION: MeshVertexAttribute = MeshVertexAttribute::new("Secondary_Position", BASE_ATTRIBUTE_ID, bevy_render::render_resource::VertexFormat::Float32x3);
pub const ATTRIBUTE_NEAR_MASK: MeshVertexAttribute = MeshVertexAttribute::new("Near_Mask", BASE_ATTRIBUTE_ID + 1, bevy_render::render_resource::VertexFormat::Uint32);
// Txz, Ty+, Ty-
pub const ATTRIBUTE_PRIMARY_TEXTURES: MeshVertexAttribute = MeshVertexAttribute::new("Side_Materials", BASE_ATTRIBUTE_ID + 2, bevy_render::render_resource::VertexFormat::Uint32x3);
// Uxz, Uy+, Uy-
pub const ATTRIBUTE_SECONDARY_TEXTURES: MeshVertexAttribute = MeshVertexAttribute::new("Vertical_Materials", BASE_ATTRIBUTE_ID + 3, bevy_render::render_resource::VertexFormat::Uint32x3);
pub const ATTRIBUTE_TEXTURE_BLEND: MeshVertexAttribute = MeshVertexAttribute::new("Material_Blend", BASE_ATTRIBUTE_ID + 4, bevy_render::render_resource::VertexFormat::Float32);

const BASE_HANDLE_ID: u64 = 2000000000000000000;
pub const VOLUMETRIC_SHADER_HANDLE: HandleUntyped = HandleUntyped::weak_from_u64(Shader::TYPE_UUID, BASE_HANDLE_ID);

pub struct VolumetricTerrainPlugin;

impl Plugin for VolumetricTerrainPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        load_internal_asset!(app, VOLUMETRIC_SHADER_HANDLE, "volumetric_terrain/volumetric_terrain.wgsl", Shader::from_wgsl);
        app.add_plugin(MaterialPlugin::<VolumetricTerrainMaterial>::default())
        .add_systems((update_terrain_lod, update_terrain_mesh_near, complete_terrain_meshes).in_base_set(RPGCoreSet::RPGUpdate))
        .add_system(sync_mesh_near_to_material.in_base_set(RPGCoreSet::RPGPostUpdate));
    }
}
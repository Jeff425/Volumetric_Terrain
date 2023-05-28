use bevy_asset::Handle;
use bevy_pbr::Material;
use bevy_render::{render_resource::AsBindGroup, prelude::Mesh, texture::Image};
use bevy_reflect::TypeUuid;

use super::{ATTRIBUTE_SECONDARY_POSITION, ATTRIBUTE_NEAR_MASK, ATTRIBUTE_PRIMARY_TEXTURES, ATTRIBUTE_SECONDARY_TEXTURES, ATTRIBUTE_TEXTURE_BLEND, VOLUMETRIC_SHADER_HANDLE};

#[derive(AsBindGroup, Debug, Clone, TypeUuid)]
#[uuid = "222113d6-e29a-4aff-8b1e-68d0a0f46847"]
pub struct VolumetricTerrainMaterial {
	// Will most likely need to create a sub-struct for this
	#[uniform(0)]
	pub mesh_near: u32,
	#[texture(1, dimension = "2d_array")]
    #[sampler(2)]
	pub texture_array: Handle<Image>,
}

impl Material for VolumetricTerrainMaterial {
	fn vertex_shader() -> bevy_render::render_resource::ShaderRef {
		//"shaders/volumetric_terrain.wgsl".into()
		VOLUMETRIC_SHADER_HANDLE.typed().into()
	}

	fn fragment_shader() -> bevy_render::render_resource::ShaderRef {
		//"shaders/volumetric_terrain.wgsl".into()
		VOLUMETRIC_SHADER_HANDLE.typed().into()
	}

	fn specialize(
			_pipeline: &bevy_pbr::MaterialPipeline<Self>,
			descriptor: &mut bevy_render::render_resource::RenderPipelineDescriptor,
			layout: &bevy_render::mesh::MeshVertexBufferLayout,
			_key: bevy_pbr::MaterialPipelineKey<Self>,
		) -> Result<(), bevy_render::render_resource::SpecializedMeshPipelineError> {
			let vertex_layout = layout.get_layout(&[
				Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
				Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
				ATTRIBUTE_SECONDARY_POSITION.at_shader_location(2),
				ATTRIBUTE_NEAR_MASK.at_shader_location(3),
				ATTRIBUTE_PRIMARY_TEXTURES.at_shader_location(4),
				ATTRIBUTE_SECONDARY_TEXTURES.at_shader_location(5),
				ATTRIBUTE_TEXTURE_BLEND.at_shader_location(6),
			])?;
			descriptor.vertex.buffers = vec![vertex_layout];
			Ok(())
	}
}
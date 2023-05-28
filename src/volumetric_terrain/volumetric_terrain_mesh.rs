use bevy_render::{prelude::Mesh, render_resource::PrimitiveTopology, mesh::Indices};
use bevy_utils::HashMap;
use glam::{Vec3, Mat3};
use std::slice::Iter;
use crate::utils::vec3i::Vec3i;

use super::{data::*, volumetric_terrain_field::{VolumetricTerrainValues, VolumetricTerrainFieldMeta}, ATTRIBUTE_SECONDARY_POSITION, ATTRIBUTE_NEAR_MASK, ATTRIBUTE_SECONDARY_TEXTURES, ATTRIBUTE_PRIMARY_TEXTURES, ATTRIBUTE_TEXTURE_BLEND};

// Used for hashing previously calculated positions to drastically reduce cache size
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, PartialOrd, Ord)]
struct CacheKey {
	pub x: usize,
	pub y: usize,
	pub z: usize,
	pub edge_code: u16,
	pub primary_texture_index: usize,
	pub secondary_texture_index: usize,
}

impl CacheKey {
	fn from_regular_cell(x: i32, y: i32, z: i32, edge_code: u16, primary_texture_index: usize, secondary_texture_index: usize) -> Self {
		Self {
			x: x as usize,
			y: y as usize,
			z: z as usize,
			edge_code,
			primary_texture_index,
			secondary_texture_index,
		}
	}

	// converts a half-resolution transition cell into a regular cell cache key
	fn uv_regular_cell(side: &TransitionSide, lod: usize, local_u: usize, local_v: usize, local_constant_val: usize, transition_reuse_data: u16,  primary_texture_index: usize, secondary_texture_index: usize) -> Self {
		let shift = (transition_reuse_data >> 4) & 0xF;
		let u = 1 + (local_u >> lod) - (shift & 0x1) as usize;
		let v = 1 + (local_v >> lod) - ((shift >> 1) & 0x1) as usize;
		let constant_s = local_constant_val >> lod;
		let u_edge = transition_reuse_data & 1 == 1;
		match side {
			LowX | HighX => Self {
				x: constant_s,
				y: u,
				z: v,
				edge_code: if u_edge {2} else {0},
				primary_texture_index,
				secondary_texture_index,
			},
			LowY | HighY => Self {
				x: u,
				y: constant_s,
				z: v,
				edge_code: if u_edge {2} else {1},
				primary_texture_index,
				secondary_texture_index,
			},
			LowZ | HighZ => Self {
				x: u,
				y: v,
				z: constant_s,
				edge_code: if u_edge {0} else {1},
				primary_texture_index,
				secondary_texture_index,
			},
		}
	}

	// Gets the transition cell cache key in xyz space. Originally was meant to share vertex across transition corners
	// But that cased some problems, so now every transition side needs to generate their own verticies
	fn uv_transition_cell(side: &TransitionSide, lod: usize, local_u: usize, local_v: usize, local_constant_val: usize, transition_reuse_data: u16,  primary_texture_index: usize, secondary_texture_index: usize) -> Self {
		let shift = (transition_reuse_data >> 4) & 0xF;
		if shift == 0x4 {
			panic!("Attempted to create a transition cache key with inside edge");
		}
		let u = 1 + (local_u >> lod) - (shift & 0x1) as usize;
		let v = 1 + (local_v >> lod) - ((shift >> 1) & 0x1) as usize;
		let constant_s = local_constant_val >> lod;
		let transition_code = transition_reuse_data & 0xF;
		let u_edge = transition_code > 4;
		let min_edge_bit = (transition_code & 1) << 2;
		let inner_edge_bit = ((shift >> 2) & 1) << 3;
		match side {
			LowX | HighX => Self {
				x: constant_s,
				y: u,
				z: v,
				edge_code: if u_edge {2} else {0} | min_edge_bit | inner_edge_bit,
				primary_texture_index,
				secondary_texture_index,
			},
			LowY | HighY => Self {
				x: u,
				y: constant_s,
				z: v,
				edge_code: if u_edge {2} else {1} | min_edge_bit | inner_edge_bit,
				primary_texture_index,
				secondary_texture_index,
			},
			LowZ | HighZ => Self {
				x: u,
				y: v,
				z: constant_s,
				edge_code: if u_edge {0} else {1} | min_edge_bit | inner_edge_bit,
				primary_texture_index,
				secondary_texture_index,
			},
		}
	}
}

// Helper function to get end index that will always allow for one extra marching cube
fn get_end_index(field_length: usize, minimum: usize, lod_gap: usize, inner_size: usize) -> usize {

	let mut end = minimum + inner_size + (lod_gap << 2);
	if end >= field_length {
		panic!("Field needs extra padding at end for this LOD. Missing {}", end - field_length + 1);
	}
	end -= lod_gap << 2; // subtract twice for central normal
	if end <= minimum {
		panic!("Block will not exist with dimension value minimum of {} and end of {}", minimum, end);
	}
	end
}

fn density_to_scale(density: f32) -> f32 {
	density / 127.0f32
}

fn get_safe_element_inc(field_length: usize, element: usize) -> usize {
	if element >= field_length - 1 {field_length - 1} else {element + 1}
}

fn get_safe_element_dec(element:usize) -> usize {
	if element > 0 {element - 1} else {element}
}

// Approximates the normal of a voxel location without having to calculate triangles
fn get_central_normal(field: &VolumetricTerrainFieldMeta, field_values: &VolumetricTerrainValues, x: usize, y: usize, z: usize) -> Vec3 {
	let field_length = field.get_sizes();
	let nx = density_to_scale(field_values.get_density(get_safe_element_inc(field_length.0, x), y, z) as f32 - field_values.get_density(get_safe_element_dec(x), y, z) as f32) * 0.5f32;
	let ny = density_to_scale(field_values.get_density(x, get_safe_element_inc(field_length.1, y), z) as f32 - field_values.get_density(x, get_safe_element_dec(y), z) as f32) * 0.5f32;
	let nz = density_to_scale(field_values.get_density(x, y, get_safe_element_inc(field_length.2, z)) as f32 - field_values.get_density(x, y, get_safe_element_dec(z)) as f32) * 0.5f32;
	Vec3::new(nx, ny, nz).normalize()
}

// Enum representings the 6 options for transition sides. Each option repsents the constant value of the side
pub enum TransitionSide {
	LowX,
	HighX,
	LowY,
	HighY,
	LowZ,
	HighZ,
}

use self::TransitionSide::*;

impl TransitionSide {

	pub fn iterator() -> Iter<'static, TransitionSide> {
		static SIDES: [TransitionSide; 6] = [LowX, HighX, LowY, HighY, LowZ, HighZ];
		SIDES.iter()
	}

	pub fn to_bit(&self) -> u32 {
		match self {
			LowX =>  0b000001,
			HighX => 0b000010,
			LowY =>  0b000100,
			HighY => 0b001000,
			LowZ =>  0b010000,
			HighZ => 0b100000,
		}
	}

	pub fn to_bit_inverse(&self) -> u32 {
		match self {
			LowX =>  0b111110,
			HighX => 0b111101,
			LowY =>  0b111011,
			HighY => 0b110111,
			LowZ =>  0b101111,
			HighZ => 0b011111,
		}
	}

	pub fn flip(&self) -> Self {
		match self {
			LowX => HighX,
			HighX => LowX,
			LowY => HighY,
			HighY => LowY,
			LowZ => HighZ,
			HighZ => LowZ,
		}
	}

	pub fn to_vec3i(&self) -> Vec3i {
		match self {
			LowX =>  Vec3i::new(-1, 0, 0),
			HighX => Vec3i::new(1, 0, 0),
			LowY =>  Vec3i::new(0, -1, 0),
			HighY => Vec3i::new(0, 1, 0),
			LowZ =>  Vec3i::new(0, 0, -1),
			HighZ => Vec3i::new(0, 0, 1),
		}
	}

	fn to_string(&self) -> &str {
		match self {
			LowX => "LowX",
			HighX => "HighX",
			LowY => "LowY",
			HighY => "HighY",
			LowZ => "LowZ",
			HighZ => "HighZ",
		}
	}

	fn get_real_inverse(&self, desired_inverse: bool) -> bool {
		match self {
			LowX | LowZ | HighY => desired_inverse,
			LowY | HighX | HighZ => !desired_inverse,
		}
	}

	fn fix_transition_max(field_length: usize, desired_max: usize, lod_gap: usize) -> usize {
		let mut end = desired_max + (lod_gap << 2);
		if end >= field_length {
			panic!("Field needs extra padding at end for this LOD. Missing {}", end - field_length + 1);
		}
		end -= lod_gap << 2; // subtract twice for central normal
		end
	}

	fn constant_val_to_local(&self, x_min: usize, y_min: usize, z_min: usize, constant_val: usize) -> usize {
		match self {
			LowX | HighX => constant_val - x_min,
			LowY | HighY => constant_val - y_min,
			LowZ | HighZ => constant_val - z_min,
		}
	}

	fn get_constant_val(&self, field_sizes: &(usize, usize, usize), x_min: usize, y_min: usize, z_min: usize, inner_size: usize, lod_gap: usize) -> usize {
		match self {
			LowX => x_min,
			LowY => y_min,
			LowZ => z_min,
			HighX => Self::fix_transition_max(field_sizes.0, x_min + inner_size, lod_gap),
			HighY => Self::fix_transition_max(field_sizes.1, y_min + inner_size, lod_gap),
			HighZ => Self::fix_transition_max(field_sizes.2, z_min + inner_size, lod_gap),
		}
	}

	// Might need to look for start values not min values. Will need the field and LOD values in that case
	fn get_min_uv(&self, x_min: usize, y_min: usize, z_min: usize) -> (usize, usize) {
		match self {
			LowX | HighX => (y_min, z_min),
			LowY | HighY => (x_min, z_min),
			LowZ | HighZ => (x_min, y_min),
		}
	}

	fn get_max_uv(&self, field_sizes: &(usize, usize, usize), x_min: usize, y_min: usize, z_min: usize, inner_size: usize, lod_gap: usize) -> (usize, usize) {
		match self {
			LowX | HighX => (Self::fix_transition_max(field_sizes.0, y_min + inner_size, lod_gap), Self::fix_transition_max(field_sizes.0, z_min + inner_size, lod_gap)),
			LowY | HighY => (Self::fix_transition_max(field_sizes.1, x_min + inner_size, lod_gap), Self::fix_transition_max(field_sizes.1, z_min + inner_size, lod_gap)),
			LowZ | HighZ => (Self::fix_transition_max(field_sizes.2, x_min + inner_size, lod_gap), Self::fix_transition_max(field_sizes.2, y_min + inner_size, lod_gap)),
		}
	}

	fn get_side_cell(&self, field_values: &VolumetricTerrainValues, u: usize, v: usize, constant_val: usize) -> i8 {
		match self {
			LowX | HighX => field_values.get_density(constant_val, u, v),
			LowY | HighY => field_values.get_density(u, constant_val, v),
			LowZ | HighZ => field_values.get_density(u, v, constant_val),
		}
	}

	fn get_side_texture(&self, field_values: &VolumetricTerrainValues, u: usize, v: usize, constant_val: usize) -> usize {
		match self {
			LowX | HighX => field_values.get_terrain_index(constant_val, u, v),
			LowY | HighY => field_values.get_terrain_index(u, constant_val, v),
			LowZ | HighZ => field_values.get_terrain_index(u, v, constant_val),
		}
	}

	fn get_position(&self, local_u: usize, local_v: usize, local_constant_val: usize) -> Vec3 {
		match self {
			LowX => Vec3::new(local_constant_val as f32, local_u as f32, local_v as f32),
			HighX => Vec3::new(local_constant_val as f32, local_u as f32, local_v as f32),
			LowY => Vec3::new(local_u as f32, local_constant_val as f32, local_v as f32),
			HighY => Vec3::new(local_u as f32, local_constant_val as f32, local_v as f32),
			LowZ => Vec3::new(local_u as f32, local_v as f32, local_constant_val as f32),
			HighZ => Vec3::new(local_u as f32, local_v as f32, local_constant_val as f32),
		}
	}

	fn get_uv_delta(vector: usize) -> (usize, usize) {
		if vector > 0x08 {
			let big_vector = vector - 0x09;
			return (big_vector & 0x1, (big_vector >> 1) & 0x1);
		}
		(vector % 3, vector / 3)
	}

	fn get_side_normal(&self, field: &VolumetricTerrainFieldMeta, field_values: &VolumetricTerrainValues, u: usize, v: usize, constant_val: usize) -> Vec3 {
		match self {
			LowX => get_central_normal(field, field_values, constant_val, u, v),
			HighX => get_central_normal(field, field_values, constant_val, u, v),
			LowY => get_central_normal(field, field_values, u, constant_val, v),
			HighY => get_central_normal(field, field_values, u, constant_val, v),
			LowZ => get_central_normal(field, field_values, u, v, constant_val),
			HighZ => get_central_normal(field, field_values, u, v, constant_val),
		}
	}

	fn extract_side_cell_texture(&self, texture_map: &mut HashMap<usize, u32>, field_values: &VolumetricTerrainValues, u: usize, v: usize, constant_val: usize) {
		if self.get_side_cell(field_values, u, v, constant_val) < 0 {
			increment_texture_count(texture_map, self.get_side_texture(field_values, u, v, constant_val));
		}
	}

	// returns the primary and secondary textures for the vertex
	fn extract_side_textures(&self, field_values: &VolumetricTerrainValues, u: usize, v: usize, constant_val: usize, lod: usize) -> (usize, usize) {
		let mut texture_map = HashMap::<usize, u32>::new();
		let lod_gap = 1<<lod;
		match self {
			LowX => extract_cell_textures(&mut texture_map, field_values, constant_val, u, v, lod),
			HighX => extract_cell_textures(&mut texture_map, field_values, constant_val - lod_gap, u, v, lod),
			LowY => extract_cell_textures(&mut texture_map, field_values, u, constant_val, v, lod),
			HighY => extract_cell_textures(&mut texture_map, field_values, u, constant_val - lod_gap, v, lod),
			LowZ => extract_cell_textures(&mut texture_map, field_values, u, v, constant_val, lod),
			HighZ => extract_cell_textures(&mut texture_map, field_values, u, v, constant_val - 1, lod),
		}
		if texture_map.len() == 0 {
			let half_gap = 1usize<<(lod-1usize);
			self.extract_side_cell_texture(&mut texture_map, field_values, u, v, constant_val);
			self.extract_side_cell_texture(&mut texture_map, field_values, u+half_gap, v, constant_val);
			self.extract_side_cell_texture(&mut texture_map, field_values, u+lod_gap, v, constant_val);
			self.extract_side_cell_texture(&mut texture_map, field_values, u, v+half_gap, constant_val);
			self.extract_side_cell_texture(&mut texture_map, field_values, u+half_gap, v+half_gap, constant_val);
			self.extract_side_cell_texture(&mut texture_map, field_values, u+lod_gap, v+half_gap, constant_val);
			self.extract_side_cell_texture(&mut texture_map, field_values, u, v+lod_gap, constant_val);
			self.extract_side_cell_texture(&mut texture_map, field_values, u+half_gap, v+lod_gap, constant_val);
			self.extract_side_cell_texture(&mut texture_map, field_values, u+lod_gap, v+lod_gap, constant_val);
		}
		if texture_map.len() == 0 {
			panic!("No textures found for side cell at {}, {} with a lod of {}", u, v, lod);
		}
		let primary_texture_index: usize;
		let secondary_texture_index: usize;
		let mut texture_count = [(300usize, 0u32); 2];
		for element in texture_map {
			if element.1 >= texture_count[0].1 {
				texture_count[1] = texture_count[0];
				texture_count[0] = element;
			} else if element.1 >= texture_count[1].1 {
				texture_count[1] = element;
			}
		}
		primary_texture_index = texture_count[0].0;
		if texture_count[1].0 == 300usize {
			secondary_texture_index = primary_texture_index;
		} else {
			secondary_texture_index = texture_count[1].0;
		}
		(primary_texture_index, secondary_texture_index)
	}

	// Returns (p0, p1, v0_u, v0_v, v1_u, v1_v, t) surface shifted if needed
	fn get_positions(
		&self, 
		field_values: &VolumetricTerrainValues, 
		constant_val: usize,
		local_constant_val: usize,
		u: usize, 
		v: usize, 
		min_u: usize, 
		min_v: usize, 
		v0: usize, 
		v1: usize, 
		d0: i16, 
		d1: i16,
		half_resolution_face: bool,
		half_resolution_lod: usize,
	) -> (Vec3, Vec3, usize, usize, usize, usize, i16) {
		let mut t = (d1 << 8) / (d1 - d0);
		let local_u = u - min_u;
		let local_v = v - min_v;
		let lod = if half_resolution_face {half_resolution_lod} else {half_resolution_lod - 1};
		let (mut v0_u, mut v0_v) = Self::get_uv_delta(v0);
		let (v1_u, v1_v) = Self::get_uv_delta(v1);
		let (v_x_u, v_x_v) = ((v1_u - v0_u) << lod, (v1_v - v0_v) << lod);
		v0_u = v0_u << lod;
		v0_v = v0_v << lod;
		let mut p0 = self.get_position(local_u + v0_u, local_v + v0_v, local_constant_val);
		let p_x = self.get_position(v_x_u, v_x_v, 0);
		let mut p1 = p0 + p_x;

		let mut real_v0_u = v0_u + u;
		let mut real_v0_v = v0_v + v;
		let mut real_v1_u = (v1_u << lod) + u;
		let mut real_v1_v = (v1_v << lod) + v;

		// Check if it is a corner (t == 0 || t == 0x100)
		if (t & 0xFF) == 0 {
			panic!("Field cannot contain 0 as a density");
		}
		let mut d0_m = d0;
		let mut d1_m = d1;
		
		// do surface shifting
		for lod_surface in (0..lod).rev() {
			// If a cell is exactly the surface, then stop early
			if d0_m == 0 || d1_m == 0 {
				break;
			}
			let back_lod = lod - lod_surface;
			let v_n_u = real_v0_u + (v_x_u >> back_lod);
			let v_n_v = real_v0_v + (v_x_v >> back_lod);
			let d_n = self.get_side_cell(field_values, v_n_u, v_n_v, constant_val) as i16;
			let p_n = p0 + (p_x * 2.0f32.powf(-(back_lod as f32)));
			if (d0_m < 0) == (d_n < 0) {
				real_v0_u = v_n_u;
				real_v0_v = v_n_v;
				d0_m = d_n;
				p0 = p_n;
			} else {
				real_v1_u = v_n_u;
				real_v1_v = v_n_v;
				d1_m = d_n;
				p1 = p_n;
			}
		}
		t = (d1_m << 8) / (d1_m - d0_m);
		(p0, p1, real_v0_u, real_v0_v, real_v1_u, real_v1_v, t)
	}

	fn extract_mesh_transition_side(
		&self,
		field: &VolumetricTerrainFieldMeta,
		field_values: &VolumetricTerrainValues,
		x_min: usize,
		y_min: usize,
		z_min: usize,
		lod: usize, // Give the LOD of the regular block we are transitioning from (the larger LOD)
		max_lod: usize,
		indices: &mut Vec<u32>,
		vertex: &mut Vec<[f32; 3]>,
		normal: &mut Vec<[f32; 3]>,
		secondary_vertex: &mut Vec<[f32; 3]>,
		
		near_mask: &mut Vec<u32>,

		primary_textures: &mut Vec<[u32; 3]>,
		secondary_textures: &mut Vec<[u32; 3]>,
		texture_blend: &mut Vec<f32>,
		vertex_cache: &mut HashMap<CacheKey, usize>,
	) {
		if lod < 1 {
			return; // Maybe panic instead
		}
		let inner_size = BLOCK_SIZE << max_lod;
	
		let lod_gap = 1usize << lod;
		let half_gap = 1usize << (lod - 1);
		
		let field_sizes = field.get_sizes();
		let constant_val = self.get_constant_val(&field_sizes, x_min, y_min, z_min, inner_size, lod_gap);
		let local_constant_val = self.constant_val_to_local(x_min, y_min, z_min, constant_val);
		let (min_u, min_v) = self.get_min_uv(x_min, y_min, z_min);
		let (max_u, max_v) = self.get_max_uv(&field_sizes, x_min, y_min, z_min, inner_size, lod_gap);
	
		// Create cache just for transition side
		// Will not be reused for other sides
		let mut transition_cache = HashMap::<CacheKey, usize>::new();

		for u in (min_u..max_u).step_by(lod_gap) {
			for v in (min_v..max_v).step_by(lod_gap) {
				let corner = [
					self.get_side_cell(field_values, u, v, constant_val),
					self.get_side_cell(field_values, u+half_gap, v, constant_val),
					self.get_side_cell(field_values, u+lod_gap, v, constant_val),
					self.get_side_cell(field_values, u, v+half_gap, constant_val),
					self.get_side_cell(field_values, u+half_gap, v+half_gap, constant_val),
					self.get_side_cell(field_values, u+lod_gap, v+half_gap, constant_val),
					self.get_side_cell(field_values, u, v+lod_gap, constant_val),
					self.get_side_cell(field_values, u+half_gap, v+lod_gap, constant_val),
					self.get_side_cell(field_values, u+lod_gap, v+lod_gap, constant_val),
					// Half resolution corners
					self.get_side_cell(field_values, u, v, constant_val),
					self.get_side_cell(field_values, u+lod_gap, v, constant_val),
					self.get_side_cell(field_values, u, v+lod_gap, constant_val),
					self.get_side_cell(field_values, u+lod_gap, v+lod_gap, constant_val),
				].map(Into::<i16>::into);
				let case_code = ((corner[0] >> 15) & 0x01)
					| ((corner[1] >> 14) & 0x02)
					| ((corner[2] >> 13) & 0x04)
					| ((corner[5] >> 12) & 0x08)
					| ((corner[8] >> 11) & 0x10)
					| ((corner[7] >> 10) & 0x20)
					| ((corner[6] >> 9) & 0x40)
					| ((corner[3] >> 8) & 0x80)
					| ((corner[4] >> 7) & 0x100);
				if case_code != 0 && case_code != 511 {
					let equivalence_class = TRANSITION_CELL_CLASS[case_code as usize];
					let inverse_winding = self.get_real_inverse(equivalence_class & 0x80 != 0);
					let cell_data = &TRANSITION_CELL_DATA[(equivalence_class & 0x7f) as usize];
					let vertex_data = &TRANSITION_VERTEX_DATA[case_code as usize];
					let vertex_count = cell_data.get_vertex_count() as usize;
					let mut vertex_map = Vec::<usize>::with_capacity(vertex_count);

					let (primary_texture_index, secondary_texture_index) = self.extract_side_textures(field_values, u, v, constant_val, lod);
					let primary_cell_textures = field.get_texture_array(primary_texture_index);
					let secondary_cell_textures = field.get_texture_array(secondary_texture_index);

					for i in 0..vertex_count {
						let transition_code = vertex_data[i];
						let reuse_data = transition_code >> 8;
						let reuse_mask = (reuse_data >> 4) & 0xF;
	
						let v0 = ((transition_code >> 4) & 0x0F) as usize;
						let v1 = (transition_code & 0x0F) as usize;

						let half_resolution_face = v0 > 0x08 || v1 > 0x08;
						// Can grab from already calculated normal cell
						if half_resolution_face {
							let cache_key = CacheKey::uv_regular_cell(self, lod, u - min_u, v - min_v, local_constant_val, reuse_data, primary_texture_index, secondary_texture_index);
							if let Some(&index) = vertex_cache.get(&cache_key) {
								vertex_map.push(index);
								continue;
							} else {
								// If a half resolution face vertex exists, then all transition meshes should have a match to a regular cell
								panic!("Transition half-face could not locate cached cell. Side: {}, ({}, {}), Real: ({}, {}) Edge: {:X?}; Returned ({}, {}, {}) edge: {}", self.to_string(), u - min_u, v - min_v, u, v, reuse_data, cache_key.x, cache_key.y, cache_key.z, cache_key.edge_code);
							}
						}
	
						let mut cache_key = None;
						if reuse_mask & 0x04 == 0 {
							// Attempt to reuse edge
							cache_key = Some(CacheKey::uv_transition_cell(self, lod, u - min_u, v - min_v, local_constant_val, reuse_data, primary_texture_index, secondary_texture_index));
							if let Some(index) = transition_cache.get(&cache_key.unwrap()) {
								vertex_map.push(*index);
								continue;
							}
						}
						let d0 = corner[v0];
						let d1 = corner[v1];
						// Check for reuse before computing the surface shifted vertex
						let (p0, p1, v0_u, v0_v, v1_u, v1_v, t) = self.get_positions(field_values, constant_val, local_constant_val, u, v, min_u, min_v, v0, v1, d0, d1, half_resolution_face, lod);
						let tf32 = (t as f32) / 256.0f32;
						let delta = 1.0f32 - tf32;
						let q = (tf32 * p0) + (delta * p1);
						let index = vertex.len();
						vertex.push(q.to_array());
						let side_normal = (self.get_side_normal(field, field_values, v0_u, v0_v, constant_val) * tf32) + (self.get_side_normal(field, field_values, v1_u, v1_v, constant_val) * delta);
						normal.push(side_normal.to_array());
						near_mask.push(0x40 | self.to_bit());
						secondary_vertex.push([0.0f32; 3]);
						primary_textures.push(primary_cell_textures);
						secondary_textures.push(secondary_cell_textures);
						let v0_index = self.get_side_texture(field_values, v0_u, v0_v, constant_val);
						let v1_index = self.get_side_texture(field_values, v1_u, v1_v, constant_val);
						texture_blend.push(calculate_texture_blend(tf32, primary_texture_index, secondary_texture_index, v0_index, v1_index));
						vertex_map.push(index);

						if let Some(key) = cache_key {
							// store for reuse
							transition_cache.insert(key, index);
						}
					}

					let triangle_count = cell_data.get_triangle_count();
					for i in (0..((triangle_count * 3) as usize)).step_by(3) {
						indices.push(vertex_map[cell_data.vertex_index[if inverse_winding {i+2} else {i}] as usize] as u32);
						indices.push(vertex_map[cell_data.vertex_index[i+1] as usize] as u32);
						indices.push(vertex_map[cell_data.vertex_index[if inverse_winding {i} else {i+2}] as usize] as u32);
					}
				}
			}
		}
	}
}

// Get the near mask of a vertex by checking if the vertex lies on any end of the mesh
fn get_near_mask(x: i32, y: i32, z: i32, inner_size: i32, v0: usize, v1: usize) -> u32 {
	let mut result = 0;
	if x == 0 && v0 & 0b1 == 0 && v1 & 0b1 == 0 {
		result |= LowX.to_bit();
	} else if x == inner_size && v0 & 0b1 == 1 && v1 & 0b1 == 1 {
		result |= HighX.to_bit();
	}
	if y == 0 && (v0 >> 1) & 0b1 == 0 && (v1 >> 1) & 0b1 == 0 {
		result |= LowY.to_bit();
	} else if y == inner_size && (v0 >> 1) & 0b1 == 1 && (v1 >> 1) & 0b1 == 1 {
		result |= HighY.to_bit();
	}
	if z == 0 && (v0 >> 2) & 0b1 == 0 && (v1 >> 2) & 0b1 == 0 {
		result |= LowZ.to_bit();
	} else if z == inner_size && (v0 >> 2) & 0b1 == 1 && (v1 >> 2) & 0b1 == 1 {
		result |= HighZ.to_bit();
	}
	result
}

fn increment_texture_count(texture_map: &mut HashMap<usize, u32>, index: usize) {
	if let Some(val) = texture_map.get_mut(&index) {
		*val += 1;
	} else {
		texture_map.insert(index, 1);
	}
}

fn determine_texture_extraction(texture_map: &mut HashMap<usize, u32>, field_values: &VolumetricTerrainValues, x: usize, y: usize, z: usize, lod: usize) {
	let corner = field_values.get_regular_corners(x, y, z, 1usize<<lod);
	let case_code = VolumetricTerrainValues::get_regular_case_code(&corner);
	if (case_code ^ (corner[7] >> 15) & 0xFF) != 0 {
		extract_cell_textures(texture_map, field_values, x, y, z, lod);
	}
}

// Recusively loops through LODs with geometry to see which cells have a texture
fn extract_cell_textures(texture_map: &mut HashMap<usize, u32>, field_values: &VolumetricTerrainValues, x: usize, y: usize, z: usize, lod: usize) {
	if lod == 0 {
		increment_texture_count(texture_map, field_values.get_terrain_index(x, y, z));
		increment_texture_count(texture_map, field_values.get_terrain_index(x+1, y, z));
		increment_texture_count(texture_map, field_values.get_terrain_index(x, y+1, z));
		increment_texture_count(texture_map, field_values.get_terrain_index(x+1, y+1, z));
		increment_texture_count(texture_map, field_values.get_terrain_index(x, y, z+1));
		increment_texture_count(texture_map, field_values.get_terrain_index(x+1, y, z+1));
		increment_texture_count(texture_map, field_values.get_terrain_index(x, y+1, z+1));
		increment_texture_count(texture_map, field_values.get_terrain_index(x+1, y+1, z+1));
	} else {
		let next_lod = lod-1;
		let lod_gap = 1usize<<(next_lod);
		determine_texture_extraction(texture_map, field_values, x, y, z, next_lod);
		determine_texture_extraction(texture_map, field_values, x+lod_gap, y, z, next_lod);
		determine_texture_extraction(texture_map, field_values, x, y+lod_gap, z, next_lod);
		determine_texture_extraction(texture_map, field_values, x+lod_gap, y+lod_gap, z, next_lod);
		determine_texture_extraction(texture_map, field_values, x, y, z+lod_gap, next_lod);
		determine_texture_extraction(texture_map, field_values, x+lod_gap, y, z+lod_gap, next_lod);
		determine_texture_extraction(texture_map, field_values, x, y+lod_gap, z+lod_gap, next_lod);
		determine_texture_extraction(texture_map, field_values, x+lod_gap, y+lod_gap, z+lod_gap, next_lod);
	}
}

fn calculate_texture_blend(t: f32, primary_texture_index: usize, secondary_texture_index: usize, v0_index: usize, v1_index: usize) -> f32 {
	let blend: f32;
	if primary_texture_index == v0_index {
		if secondary_texture_index == v1_index {
			blend = t;
		} else {
			blend = 1.0;
		}
	} else if primary_texture_index == v1_index {
		if secondary_texture_index == v0_index {
			blend = 1.0f32 - t;
		} else {
			blend = 1.0;
		}
	} else if secondary_texture_index == v0_index || secondary_texture_index == v1_index {
		blend = 0.0;
	} else {
		blend = 1.0;
	}
	blend
}

// Extracts the mesh for a voxel terrain field, using the given minimum coordinates for the starting point
// Size of the mesh is 16 * 2^max_lod per dimension
// Computes 16 * 2^(max_lod - lod) cells per dimension
// For any lod > 1, will also generate transition sides to go one lod lower
pub fn extract_volumetric_terrain_mesh(
	field: &VolumetricTerrainFieldMeta,
	field_values: &VolumetricTerrainValues,
	x_min: usize,
	y_min: usize,
	z_min: usize,
	lod: usize,
	max_lod: usize,
) -> Mesh {
	// The gap between cells due to LOD. LOD_0 only separated by 1
	let lod_gap = 1usize << lod;
	
	let inner_size = BLOCK_SIZE << (max_lod - lod);

	// Vertex of mesh
	let mut vertex = Vec::<[f32; 3]>::new();
	let mut normal = Vec::<[f32; 3]>::new();
	let mut secondary_vertex = Vec::<[f32; 3]>::new();
	let mut primary_textures = Vec::<[u32; 3]>::new();
	let mut secondary_textures = Vec::<[u32; 3]>::new();
	let mut texture_blend = Vec::<f32>::new();
	let mut near_mask = Vec::<u32>::new();
	// 1 111111
	// Maximal bit is for determining what to do when next to a lower LOD cell that matches the side (uniform will be passed in)
	// 6 end bits determine which edges vertex is on
	// 0 = No Match: use primary, Match: use secondary. 1 = No Match: Delete, Match: use primary

	// Owned verticies for each cell, for reusing
	// Using a hashmap to be significantly more cost-effective

	let mut cell_owned_vertex = HashMap::<CacheKey, usize>::new();
	let mut indices = Vec::<u32>::new();

	let local_x_min = x_min >> lod;
	let local_y_min = y_min >> lod;
	let local_z_min = z_min >> lod;

	let x_start = x_min;
	let y_start = y_min;
	let z_start = z_min;

	let full_inner_size = BLOCK_SIZE << max_lod;
	let field_sizes = field.get_sizes();
	let x_end = get_end_index(field_sizes.0, x_min, lod_gap, full_inner_size);
	let y_end = get_end_index(field_sizes.1, y_min, lod_gap, full_inner_size);
	let z_end = get_end_index(field_sizes.2, z_min, lod_gap, full_inner_size);

	// Used to store the index of any vertex that borders the block
	let mut transition_verticies = Vec::<usize>::new();

	for x in (x_start..x_end).step_by(lod_gap) {
		for y in (y_start..y_end).step_by(lod_gap) {
			for z in (z_start..z_end).step_by(lod_gap) {
				// If start and end were calculated correctly for all dimensions, then every value here will exist
				let corner = field_values.get_regular_corners(x, y, z, lod_gap);
				let case_code = VolumetricTerrainValues::get_regular_case_code(&corner);
				if (case_code ^ (corner[7] >> 15) & 0xFF) != 0 {
					let case_code_u = case_code as usize;
					let cell_data = &REGULAR_CELL_DATA[REGULAR_CELL_CLASS[case_code_u] as usize];
					let vertex_data = &REGULAR_VERTEX_DATA[case_code_u];

					let local_x = (x >> lod) as i32 - local_x_min as i32;
					let local_y = (y >> lod) as i32 - local_y_min as i32;
					let local_z = (z >> lod) as i32 - local_z_min as i32;

					// Calculate the primary and secondary textures
					let primary_texture_index: usize;
					let secondary_texture_index: usize;
					{
						let mut texture_map = HashMap::<usize, u32>::new();
						extract_cell_textures(&mut texture_map, field_values, x, y, z, lod);
						let mut texture_count = [(300usize, 0u32); 2];
						for element in texture_map {
							if element.1 >= texture_count[0].1 {
								texture_count[1] = texture_count[0];
								texture_count[0] = element;
							} else if element.1 >= texture_count[1].1 {
								texture_count[1] = element;
							}
						}
						primary_texture_index = texture_count[0].0;
						if texture_count[1].0 == 300usize {
							secondary_texture_index = primary_texture_index;
						} else {
							secondary_texture_index = texture_count[1].0;
						}
					}

					let primary_cell_textures = field.get_texture_array(primary_texture_index);
					let secondary_cell_textures = field.get_texture_array(secondary_texture_index);
					
					// Stores the vertex for the triangles in either an index format or real format
					let mut vertex_map = Vec::<usize>::new();
					for i in 0..(cell_data.get_vertex_count() as usize) {

						let edge_code = vertex_data[i];
						// first figure out if vertex already exists
						let shift_code = edge_code >> 12;
						let edge_position = ((edge_code >> 8) & 0xF) - 1;
						let x_shift = (shift_code & 0b1) as i32;
						let y_shift = ((shift_code & 0b10) >> 1) as i32;
						let z_shift = ((shift_code & 0b100) >> 2) as i32;

						// If any shift variable does not equal 0 then this vertex should already be calculated
						// But it is not garunteed if vertex is adjacent to a field edge
						let prev_x = 1 + local_x - x_shift;
						let prev_y = 1 + local_y - y_shift;
						let prev_z = 1 + local_z - z_shift;

						let cache_key = CacheKey::from_regular_cell(prev_x, prev_y, prev_z, edge_position, primary_texture_index, secondary_texture_index);
						if let Some(owned_index) = cell_owned_vertex.get(&cache_key) {
							// Check if vertex has already been calculated and shares the same textures
							vertex_map.push(*owned_index);
							continue;
						}

						let v0 = ((edge_code >> 4) & 0x0F) as usize;
						let v1 = (edge_code & 0x0F) as usize;
						let v_x = v0 ^ v1;
						
						let mut d0 = corner[v0];
						let mut d1 = corner[v1];

						let cell_size = lod_gap as f32;
						let mut p0 = Vec3::new((local_x + (v0 & 0b1) as i32) as f32, (local_y + ((v0 >> 1) & 0b1) as i32) as f32, (local_z + ((v0 >> 2) & 0b1) as i32) as f32) * cell_size;
						let p_x = Vec3::new((v_x & 0b1) as f32, ((v_x >> 1) & 0b1) as f32, ((v_x >> 2) & 0b1) as f32);
						let mut p1 = p0 + (p_x * cell_size);

						let mut t = (d1 << 8) / (d1 - d0);

						if (t & 0x00FF) != 0 {
							// Vertex lies in the interior of the edge
							// first surface shift
							let mut v0_x = x + ((v0 & 0b1) << lod);
							let mut v0_y = y + (((v0 >> 1) & 0b1) << lod);
							let mut v0_z = z + (((v0 >> 2) & 0b1) << lod);
							let mut v1_x = x + ((v1 & 0b1) << lod);
							let mut v1_y = y + (((v1 >> 1) & 0b1) << lod);
							let mut v1_z = z + (((v1 >> 2) & 0b1) << lod);
							for lod_surface in (0..lod).rev() {
								// If a cell is exactly the surface, then stop early
								if d0 == 0 || d1 == 0 {
									break;
								}
								let v_n_x = v0_x + ((v_x & 0b1) << lod_surface);
								let v_n_y = v0_y + (((v_x >> 1) & 0b1) << lod_surface);
								let v_n_z = v0_z + (((v_x >> 2) & 0b1) << lod_surface);
								let d_n = field_values.get_density(v_n_x, v_n_y, v_n_z) as i16;
								let surface_cell_size = (1 << lod_surface) as f32;
								let p_n = p0 + (p_x * surface_cell_size);
								if (d0 < 0) == (d_n < 0) {
									v0_x = v_n_x;
									v0_y = v_n_y;
									v0_z = v_n_z;
									d0 = d_n;
									p0 = p_n;
								} else {
									// if d_n == 0 then will use p1 always as the replacement
									d1 = d_n;
									p1 = p_n;
									v1_x = v_n_x;
									v1_y = v_n_y;
									v1_z = v_n_z;
								}
							}
							t = (d1 << 8) / (d1 - d0);
							let tf32 = (t as f32) / 256.0f32;
							let u = 1.0f32 - tf32;
							
							let q = (tf32 * p0) + (u * p1);

							let index = vertex.len();
							vertex.push([q.x, q.y, q.z]);
							let vert_norm = (get_central_normal(field, field_values, v0_x, v0_y, v0_z) * tf32) + (get_central_normal(field, field_values, v1_x, v1_y, v1_z) * u);
							normal.push(vert_norm.to_array());
							secondary_vertex.push([0.0f32; 3]);
							if lod > 0 {
								let local_near_mask = get_near_mask(prev_x, prev_y, prev_z, inner_size as i32, v0, v1);
								near_mask.push(local_near_mask);
								if local_near_mask > 0 {
									transition_verticies.push(index);
								}
							} else {
								near_mask.push(0);
							}
							primary_textures.push(primary_cell_textures);
							secondary_textures.push(secondary_cell_textures);
							let v0_index = field_values.get_terrain_index(v0_x, v0_y, v0_z);
							let v1_index = field_values.get_terrain_index(v1_x, v1_y, v1_z);
							texture_blend.push(calculate_texture_blend(tf32, primary_texture_index, secondary_texture_index, v0_index, v1_index));
							cell_owned_vertex.insert(cache_key, index);
							vertex_map.push(index);
						} else {
							// Do not allow 0 density values
							panic!("Field cannot contain 0 as a density");
						}
					}
					let triangle_count = cell_data.get_triangle_count();
					for i in (0..((triangle_count * 3) as usize)).step_by(3) {
						let index0 = vertex_map[cell_data.vertex_index[i] as usize];
						let index1 = vertex_map[cell_data.vertex_index[i+1] as usize];
						let index2 = vertex_map[cell_data.vertex_index[i+2] as usize];
						let p0 = Vec3::from(vertex[index0]);
						let p1 = Vec3::from(vertex[index1]);
						let p2 = Vec3::from(vertex[index2]);
						if (p1 - p0).cross(p2 - p0).try_normalize().is_some() {
							// Only add the triangle if the normal can exist
							indices.push(index0 as u32);
							indices.push(index1 as u32);
							indices.push(index2 as u32);
						}
					}
				}
			}
		}
	}

	// if needed to make normals unit normals
	normal.iter_mut().for_each(|n| *n = Vec3::from(*n).normalize_or_zero().to_array());

	// Need to shift all secondary positions transitional verticies
	// k == lod, 2^k == lod_gap, s == inner_size
	let p2k = lod_gap as f32;
	let wk = 2f32.powf(lod as f32 - 2.0f32);
	let p2mk = 2f32.powf(-(lod as f32));
	let s = inner_size as f32;
	for index in transition_verticies {
		let mut delta_arr = [0.0f32; 3];
		for i in 0usize..3 {
			let v = vertex[index][i];
			if v < p2k {
				delta_arr[i] = (1.0f32 - (p2mk * v)) * wk;
			} else if v > (p2k * (s - 1.0f32)) {
				delta_arr[i] = (s - 1.0f32 - (p2mk * v)) * wk;
			}
		}
		let n = Vec3::from_array(normal[index]);
		// Column-Major order for the normal projection
		let mat = Mat3::from_cols_array(&[
			1.0f32 - (n.x * n.x), -n.x * n.y, -n.x * n.z,
			-n.x * n.y, 1.0f32 - (n.y * n.y), -n.y * n.z,
			-n.x * n.z, -n.y * n.z, 1.0f32 - (n.z * n.z)
		]);
		secondary_vertex[index] = (Vec3::from_array(vertex[index]) + (mat * Vec3::from_array(delta_arr))).to_array();
	}

	// Next is to create the transitional faces, passing in the cell_owned_vertex to grab the already generated vertex indicies
	if lod > 0 {
		for side in TransitionSide::iterator() {
			side.extract_mesh_transition_side(
				field, 
				field_values,
				x_min, 
				y_min, 
				z_min, 
				lod, 
				max_lod, 
				&mut indices, 
				&mut vertex, 
				&mut normal, 
				&mut secondary_vertex, 
				&mut near_mask, 
				&mut primary_textures, 
				&mut secondary_textures, 
				&mut texture_blend, 
				&mut cell_owned_vertex
			);
		}
	}
	let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
	mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertex);
	mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normal);
	mesh.insert_attribute(ATTRIBUTE_SECONDARY_POSITION, secondary_vertex);
	mesh.insert_attribute(ATTRIBUTE_NEAR_MASK, near_mask);
	mesh.insert_attribute(ATTRIBUTE_PRIMARY_TEXTURES, primary_textures);
	mesh.insert_attribute(ATTRIBUTE_SECONDARY_TEXTURES, secondary_textures);
	mesh.insert_attribute(ATTRIBUTE_TEXTURE_BLEND, texture_blend);
	mesh.set_indices(Some(Indices::U32(indices)));
	mesh
}
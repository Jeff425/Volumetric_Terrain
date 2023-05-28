//#import bevy_pbr::mesh_view_types
#import bevy_pbr::mesh_view_bindings
#import bevy_pbr::pbr_bindings
#import bevy_pbr::mesh_bindings
#import bevy_pbr::utils
#import bevy_pbr::clustered_forward
#import bevy_pbr::lighting
#import bevy_pbr::shadows
#import bevy_pbr::fog
#import bevy_pbr::pbr_ambient
#import bevy_pbr::pbr_functions
#import bevy_pbr::mesh_functions

struct VolumetricTerrainMaterial {
    mesh_near: u32,
};
@group(1) @binding(0)
var<uniform> v_material: VolumetricTerrainMaterial;
@group(1) @binding(1)
var texture_array: texture_2d_array<f32>;
@group(1) @binding(2)
var texture_sampler: sampler;

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) secondary_position: vec3<f32>,
    @location(3) near_mask: u32,
    @location(4) primary_textures: vec3<u32>,
    @location(5) secondary_textures: vec3<u32>,
    @location(6) texture_blend: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) remove_frag: f32,
    @location(2) world_position: vec4<f32>,
    @location(3) primary_textures: vec3<u32>,
    @location(4) secondary_textures: vec3<u32>,
    @location(5) texture_blend: f32,
};

struct FragmentInput {
    //@builtin(front_facing) is_front: bool,
    @builtin(position) frag_coord: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) remove_frag: f32,
    @location(2) world_position: vec4<f32>,
    @location(3) primary_textures: vec3<u32>,
    @location(4) secondary_textures: vec3<u32>,
    @location(5) texture_blend: f32,
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    out.normal = vertex.normal;
    if (vertex.near_mask & 0x40u) != 0u {
        out.world_position = vec4<f32>(vertex.position, 1.0);
        // Transition masks will never be 0. Make sure the material includes all side the transition wants
        if (vertex.near_mask & (v_material.mesh_near | 0x40u)) == vertex.near_mask {
            out.remove_frag = 0.0;
        } else {
            out.remove_frag = 1.0;
        }
    } else {
        out.remove_frag = 0.0;
        // Mask can be 0 so do a check for that. Then check that all sides match the material before using secondary vertex
        if vertex.near_mask != 0u && (vertex.near_mask & v_material.mesh_near) == vertex.near_mask {
            out.world_position = vec4<f32>(vertex.secondary_position, 1.0);
        } else {
            out.world_position = vec4<f32>(vertex.position, 1.0);
        }
    }
    out.world_position = mesh_position_local_to_world(mesh.model, out.world_position);
    out.clip_position = mesh_position_world_to_clip(out.world_position);
    out.primary_textures = vertex.primary_textures;
    out.secondary_textures = vertex.secondary_textures;
    out.texture_blend = vertex.texture_blend;
    return out;
}

@fragment
fn fragment(input: FragmentInput) -> @location(0) vec4<f32> {
    var blend = clamp(abs(normalize(input.normal)) - 0.5, vec3<f32>(0.0), vec3<f32>(1.0, 1.0, 1.0));
    blend *= blend;
    blend *= blend;
    blend *= blend;
    blend = blend / dot(blend, vec3<f32>(1.0, 1.0, 1.0));
    var flip = vec3<f32>(f32(input.normal.x < 0.0), f32(input.normal.z >= 0.0), f32(input.normal.y < 0.0));
    var vert_materials = select(vec2<u32>(input.primary_textures.y, input.secondary_textures.y), vec2<u32>(input.primary_textures.z, input.secondary_textures.z), input.normal.y < 0.0);
    var side_coord = vec3<f32>(mix(input.world_position.zx, -input.world_position.zx, flip.xy), input.world_position.y);
    var vert_coord = vec2<f32>(mix(input.world_position.x, -input.world_position.x, flip.z), input.world_position.z);

    var x_primary = textureSample(texture_array, texture_sampler, side_coord.xz, i32(input.primary_textures.x));
    var z_primary = textureSample(texture_array, texture_sampler, side_coord.yz, i32(input.primary_textures.x));
    var y_primary = textureSample(texture_array, texture_sampler, vert_coord, i32(vert_materials.x));

    var x_secondary = textureSample(texture_array, texture_sampler, side_coord.xz, i32(input.secondary_textures.x));
    var z_secondary = textureSample(texture_array, texture_sampler, side_coord.yz, i32(input.secondary_textures.x));
    var y_secondary = textureSample(texture_array, texture_sampler, vert_coord, i32(vert_materials.y));

    if input.remove_frag > 0.0 {
        discard;
    }
    var color1 = (blend.x * x_primary) + (blend.y * y_primary) + (blend.z * z_primary);
    var color2 = (blend.x * x_secondary) + (blend.y * y_secondary) + (blend.z * z_secondary);
    // Sine
    var texture_blend = -(cos(3.14159 * input.texture_blend) - 1.0) / 2.0;
    //return vec4<f32>((input.normal + vec3<f32>(1.0)) * 0.5, 1.0);
    //return mix(color2, color1, texture_blend);

    // Use the mixed color to setup the pbr pipeline
    // Currently assigning defaults from https://github.com/bevyengine/bevy/blob/main/crates/bevy_pbr/src/pbr_material.rs
    // Might want to look into sampling reflective maps, emissive maps, occlusion maps, metallic uniforms etc
    var pbr_input = pbr_input_new();
    pbr_input.material.base_color = mix(color2, color1, texture_blend);
    pbr_input.frag_coord = input.frag_coord;
    pbr_input.world_position = input.world_position;
    pbr_input.world_normal = input.normal;
    pbr_input.is_orthographic = view.projection[3].w == 1.0;
    // Skip normal mapping since there are no support for any normal maps
    pbr_input.N = normalize(pbr_input.world_normal);
    pbr_input.V = calculate_view(input.world_position, pbr_input.is_orthographic);
    var output_color = pbr(pbr_input);

    if (fog.mode != FOG_MODE_OFF) {
        output_color = apply_fog(output_color, input.world_position.xyz, view.world_position.xyz);
    }
    
    // TONEMAPPING
    output_color = tone_mapping(output_color);

    // DEBAND DITHER
    var output_rgb = output_color.rgb;
    output_rgb = pow(output_rgb, vec3<f32>(1.0 / 2.2));
    output_rgb = output_rgb + screen_space_dither(input.frag_coord.xy);
    // This conversion back to linear space is required because our output texture format is
    // SRGB; the GPU will assume our output is linear and will apply an SRGB conversion.
    output_rgb = pow(output_rgb, vec3<f32>(2.2));
    output_color = vec4(output_rgb, output_color.a);

    return output_color;
}
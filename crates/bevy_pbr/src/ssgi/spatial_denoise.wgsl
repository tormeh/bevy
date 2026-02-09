// 3x3 bilateral filter (edge-preserving blur) for SSGI
// https://people.csail.mit.edu/sparis/bf_course/course_notes.pdf
//
// Denoises both the ambient occlusion (R channel) and indirect light (GBA channels)
// simultaneously using the same edge-preserving weights derived from depth differences.
//
// Note: Does not use the Gaussian kernel part of a typical bilateral blur
// From the paper: "use the information gathered on a neighborhood of 4 × 4 using a bilateral filter for
// reconstruction, using _uniform_ convolution weights"

#import bevy_render::view::View

@group(0) @binding(0) var ssgi_noisy: texture_2d<f32>;
@group(0) @binding(1) var depth_differences: texture_2d<u32>;
@group(0) @binding(2) var ssgi_denoised: texture_storage_2d<rgba16float, write>;
@group(1) @binding(0) var point_clamp_sampler: sampler;
@group(1) @binding(1) var linear_clamp_sampler: sampler;
@group(1) @binding(2) var<uniform> view: View;

// Safely load from the noisy SSGI texture, clamping to valid coordinates.
fn load_ssgi(coords: vec2<i32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(ssgi_noisy));
    let clamped = clamp(coords, vec2<i32>(0), dims - vec2<i32>(1));
    return textureLoad(ssgi_noisy, clamped, 0);
}

@compute
@workgroup_size(8, 8, 1)
fn spatial_denoise(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_coordinates = vec2<i32>(global_id.xy);
    let uv = vec2<f32>(pixel_coordinates) / view.viewport.zw;

    // Load edge weights from depth differences using textureGather (single-channel texture)
    let edges0 = textureGather(0, depth_differences, point_clamp_sampler, uv);
    let edges1 = textureGather(0, depth_differences, point_clamp_sampler, uv, vec2<i32>(2i, 0i));
    let edges2 = textureGather(0, depth_differences, point_clamp_sampler, uv, vec2<i32>(1i, 2i));

    let left_edges = unpack4x8unorm(edges0.x);
    let right_edges = unpack4x8unorm(edges1.x);
    let top_edges = unpack4x8unorm(edges0.z);
    let bottom_edges = unpack4x8unorm(edges2.w);
    var center_edges = unpack4x8unorm(edges0.y);
    center_edges *= vec4<f32>(left_edges.y, right_edges.x, top_edges.w, bottom_edges.z);

    let center_weight = 1.2;
    let left_weight = center_edges.x;
    let right_weight = center_edges.y;
    let top_weight = center_edges.z;
    let bottom_weight = center_edges.w;
    let top_left_weight = 0.425 * (top_weight * top_edges.x + left_weight * left_edges.z);
    let top_right_weight = 0.425 * (top_weight * top_edges.y + right_weight * right_edges.z);
    let bottom_left_weight = 0.425 * (bottom_weight * bottom_edges.x + left_weight * left_edges.w);
    let bottom_right_weight = 0.425 * (bottom_weight * bottom_edges.y + right_weight * right_edges.w);

    // Load all 9 neighbors using textureLoad (reads all RGBA channels at once)
    let center = load_ssgi(pixel_coordinates);
    let left = load_ssgi(pixel_coordinates + vec2<i32>(-1, 0));
    let right = load_ssgi(pixel_coordinates + vec2<i32>(1, 0));
    let top = load_ssgi(pixel_coordinates + vec2<i32>(0, -1));
    let bottom = load_ssgi(pixel_coordinates + vec2<i32>(0, 1));
    let top_left = load_ssgi(pixel_coordinates + vec2<i32>(-1, -1));
    let top_right = load_ssgi(pixel_coordinates + vec2<i32>(1, -1));
    let bottom_left = load_ssgi(pixel_coordinates + vec2<i32>(-1, 1));
    let bottom_right = load_ssgi(pixel_coordinates + vec2<i32>(1, 1));

    // Weighted sum of all channels (AO in R, indirect light in GBA)
    var sum = center * center_weight;
    sum += left * left_weight;
    sum += right * right_weight;
    sum += top * top_weight;
    sum += bottom * bottom_weight;
    sum += top_left * top_left_weight;
    sum += top_right * top_right_weight;
    sum += bottom_left * bottom_left_weight;
    sum += bottom_right * bottom_right_weight;

    var sum_weight = center_weight;
    sum_weight += left_weight;
    sum_weight += right_weight;
    sum_weight += top_weight;
    sum_weight += bottom_weight;
    sum_weight += top_left_weight;
    sum_weight += top_right_weight;
    sum_weight += bottom_left_weight;
    sum_weight += bottom_right_weight;

    let denoised = sum / sum_weight;

    textureStore(ssgi_denoised, pixel_coordinates, denoised);
}

use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, Handle};
use bevy_camera::{Camera, Camera3d};
use bevy_core_pipeline::{
    prepass::{DeferredPrepass, DepthPrepass, NormalPrepass, ViewPrepassTextures},
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::{
    prelude::{Component, Entity},
    query::{Has, With},
    reflect::ReflectComponent,
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Commands, Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_image::ToExtents;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    camera::{ExtractedCamera, TemporalJitter},
    diagnostic::RecordDiagnostics,
    extract_component::ExtractComponent,
    globals::{GlobalsBuffer, GlobalsUniform},
    render_resource::{
        binding_types::{
            sampler, texture_2d, texture_depth_2d, texture_storage_2d, uniform_buffer,
        },
        *,
    },
    renderer::{RenderAdapter, RenderContext, RenderDevice, RenderQueue, ViewQuery},
    sync_component::SyncComponentPlugin,
    sync_world::RenderEntity,
    texture::{CachedTexture, TextureCache},
    view::{Msaa, ViewUniform, ViewUniformOffset, ViewUniforms},
    Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
};
use bevy_shader::{load_shader_library, Shader, ShaderDefVal};
use bevy_utils::prelude::default;
use core::mem;
use tracing::{error, warn};

/// Plugin for screen space global illumination.
pub struct ScreenSpaceGlobalIlluminationPlugin;

impl Plugin for ScreenSpaceGlobalIlluminationPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "ssgi_utils.wgsl");

        embedded_asset!(app, "preprocess_depth.wgsl");
        embedded_asset!(app, "ssgi.wgsl");
        embedded_asset!(app, "spatial_denoise.wgsl");

        app.add_plugins(SyncComponentPlugin::<ScreenSpaceGlobalIllumination>::default());
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        if render_app
            .world()
            .resource::<RenderDevice>()
            .limits()
            .max_storage_textures_per_shader_stage
            < 5
        {
            warn!("ScreenSpaceGlobalIlluminationPlugin not loaded. GPU lacks support: Limits::max_storage_textures_per_shader_stage is less than 5.");
            return;
        }

        render_app
            .init_resource::<SsgiPipelines>()
            .init_resource::<SpecializedComputePipelines<SsgiPipelines>>()
            .add_systems(ExtractSchedule, extract_ssgi_settings)
            .add_systems(
                Render,
                (
                    prepare_ssgi_pipelines.in_set(RenderSystems::Prepare),
                    prepare_ssgi_textures.in_set(RenderSystems::PrepareResources),
                    prepare_ssgi_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            );

        render_app.add_systems(
            Core3d,
            ssgi.after(Core3dSystems::Prepass)
                .before(Core3dSystems::MainPass),
        );
    }
}

/// Component to apply screen space global illumination to a 3d camera.
///
/// Screen space global illumination (SSGI) approximates indirect diffuse
/// lighting by sampling nearby surface colors from the deferred GBuffer during
/// horizon-based ray marching. It also computes ambient occlusion as a byproduct.
///
/// # Usage Notes
///
/// Requires that you add [`ScreenSpaceGlobalIlluminationPlugin`] to your app,
/// and that the camera uses deferred rendering ([`DeferredPrepass`]).
///
/// It is strongly recommended that you use SSGI in conjunction with
/// TAA (`TemporalAntiAliasing`).
/// Doing so greatly reduces SSGI noise.
///
/// SSGI is not supported on `WebGL2`, and is not currently supported on `WebGPU`.
#[derive(Component, ExtractComponent, Reflect, PartialEq, Clone, Debug)]
#[reflect(Component, Debug, Default, PartialEq, Clone)]
#[require(DepthPrepass, NormalPrepass, DeferredPrepass)]
#[doc(alias = "Ssgi")]
pub struct ScreenSpaceGlobalIllumination {
    /// Quality of the SSGI effect.
    pub quality_level: ScreenSpaceGlobalIlluminationQualityLevel,
    /// A constant estimated thickness of objects.
    ///
    /// This value is used to decide how far behind an object a ray of light needs to be in order
    /// to pass behind it. Any ray closer than that will be occluded.
    pub constant_object_thickness: f32,
}

impl Default for ScreenSpaceGlobalIllumination {
    fn default() -> Self {
        Self {
            quality_level: ScreenSpaceGlobalIlluminationQualityLevel::default(),
            constant_object_thickness: 0.25,
        }
    }
}

#[derive(Reflect, PartialEq, Eq, Hash, Clone, Copy, Default, Debug)]
#[reflect(PartialEq, Hash, Clone, Default)]
pub enum ScreenSpaceGlobalIlluminationQualityLevel {
    Low,
    Medium,
    #[default]
    High,
    Ultra,
    Custom {
        /// Higher slice count means less noise, but worse performance.
        slice_count: u32,
        /// Samples per slice side is also tweakable, but recommended to be left at 2 or 3.
        samples_per_slice_side: u32,
    },
}

impl ScreenSpaceGlobalIlluminationQualityLevel {
    fn sample_counts(&self) -> (u32, u32) {
        match self {
            Self::Low => (1, 2),    // 4 spp (1 * (2 * 2)), plus optional temporal samples
            Self::Medium => (2, 2), // 8 spp (2 * (2 * 2)), plus optional temporal samples
            Self::High => (3, 3),   // 18 spp (3 * (3 * 2)), plus optional temporal samples
            Self::Ultra => (9, 3),  // 54 spp (9 * (3 * 2)), plus optional temporal samples
            Self::Custom {
                slice_count: slices,
                samples_per_slice_side,
            } => (*slices, *samples_per_slice_side),
        }
    }
}

// The SSGI output format: R = AO visibility, GBA = indirect light RGB.
const SSGI_OUTPUT_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

fn ssgi(
    view: ViewQuery<(
        &ExtractedCamera,
        &SsgiPipelineId,
        &SsgiBindGroups,
        &ViewUniformOffset,
    )>,
    pipelines: Res<SsgiPipelines>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let (camera, pipeline_id, bind_groups, view_uniform_offset) = view.into_inner();

    let (
        Some(camera_size),
        Some(preprocess_depth_pipeline),
        Some(spatial_denoise_pipeline),
        Some(ssgi_pipeline),
    ) = (
        camera.physical_viewport_size,
        pipeline_cache.get_compute_pipeline(pipelines.preprocess_depth_pipeline),
        pipeline_cache.get_compute_pipeline(pipelines.spatial_denoise_pipeline),
        pipeline_cache.get_compute_pipeline(pipeline_id.0),
    )
    else {
        return;
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let time_span = diagnostics.time_span(ctx.command_encoder(), "ssgi");

    let command_encoder = ctx.command_encoder();
    command_encoder.push_debug_group("ssgi");

    {
        let mut preprocess_depth_pass =
            command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("ssgi_preprocess_depth"),
                timestamp_writes: None,
            });
        preprocess_depth_pass.set_pipeline(preprocess_depth_pipeline);
        preprocess_depth_pass.set_bind_group(0, &bind_groups.preprocess_depth_bind_group, &[]);
        preprocess_depth_pass.set_bind_group(
            1,
            &bind_groups.common_bind_group,
            &[view_uniform_offset.offset],
        );
        preprocess_depth_pass.dispatch_workgroups(
            camera_size.x.div_ceil(16),
            camera_size.y.div_ceil(16),
            1,
        );
    }

    {
        let mut ssgi_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("ssgi"),
            timestamp_writes: None,
        });
        ssgi_pass.set_pipeline(ssgi_pipeline);
        ssgi_pass.set_bind_group(0, &bind_groups.ssgi_bind_group, &[]);
        ssgi_pass.set_bind_group(
            1,
            &bind_groups.common_bind_group,
            &[view_uniform_offset.offset],
        );
        ssgi_pass.dispatch_workgroups(camera_size.x.div_ceil(8), camera_size.y.div_ceil(8), 1);
    }

    {
        let mut spatial_denoise_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("ssgi_spatial_denoise"),
            timestamp_writes: None,
        });
        spatial_denoise_pass.set_pipeline(spatial_denoise_pipeline);
        spatial_denoise_pass.set_bind_group(0, &bind_groups.spatial_denoise_bind_group, &[]);
        spatial_denoise_pass.set_bind_group(
            1,
            &bind_groups.common_bind_group,
            &[view_uniform_offset.offset],
        );
        spatial_denoise_pass.dispatch_workgroups(
            camera_size.x.div_ceil(8),
            camera_size.y.div_ceil(8),
            1,
        );
    }

    command_encoder.pop_debug_group();
    time_span.end(ctx.command_encoder());
}

#[derive(Resource)]
struct SsgiPipelines {
    preprocess_depth_pipeline: CachedComputePipelineId,
    spatial_denoise_pipeline: CachedComputePipelineId,

    common_bind_group_layout: BindGroupLayoutDescriptor,
    preprocess_depth_bind_group_layout: BindGroupLayoutDescriptor,
    ssgi_bind_group_layout: BindGroupLayoutDescriptor,
    spatial_denoise_bind_group_layout: BindGroupLayoutDescriptor,

    hilbert_index_lut: TextureView,
    point_clamp_sampler: Sampler,
    linear_clamp_sampler: Sampler,

    shader: Handle<Shader>,
    depth_format: TextureFormat,
}

impl FromWorld for SsgiPipelines {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let render_queue = world.resource::<RenderQueue>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // Detect the depth format support for the hierarchical depth mip chain
        let render_adapter = world.resource::<RenderAdapter>();
        let depth_format = if render_adapter
            .get_texture_format_features(TextureFormat::R16Float)
            .allowed_usages
            .contains(TextureUsages::STORAGE_BINDING)
        {
            TextureFormat::R16Float
        } else {
            TextureFormat::R32Float
        };

        let hilbert_index_lut = render_device
            .create_texture_with_data(
                render_queue,
                &(TextureDescriptor {
                    label: Some("ssgi_hilbert_index_lut"),
                    size: Extent3d {
                        width: HILBERT_WIDTH as u32,
                        height: HILBERT_WIDTH as u32,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::R16Uint,
                    usage: TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                }),
                TextureDataOrder::default(),
                bytemuck::cast_slice(&generate_hilbert_index_lut()),
            )
            .create_view(&TextureViewDescriptor::default());

        let point_clamp_sampler = render_device.create_sampler(&SamplerDescriptor {
            min_filter: FilterMode::Nearest,
            mag_filter: FilterMode::Nearest,
            mipmap_filter: MipmapFilterMode::Nearest,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            ..Default::default()
        });
        let linear_clamp_sampler = render_device.create_sampler(&SamplerDescriptor {
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Nearest,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            ..Default::default()
        });

        let common_bind_group_layout = BindGroupLayoutDescriptor::new(
            "ssgi_common_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    sampler(SamplerBindingType::NonFiltering),
                    sampler(SamplerBindingType::Filtering),
                    uniform_buffer::<ViewUniform>(true),
                ),
            ),
        );

        let preprocess_depth_bind_group_layout = BindGroupLayoutDescriptor::new(
            "ssgi_preprocess_depth_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_depth_2d(),
                    texture_storage_2d(depth_format, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(depth_format, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(depth_format, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(depth_format, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(depth_format, StorageTextureAccess::WriteOnly),
                ),
            ),
        );

        // SSGI bind group: reads preprocessed depth, normals, hilbert LUT, deferred GBuffer;
        // writes SSGI output (RGBA16Float) and depth differences.
        let ssgi_bind_group_layout = BindGroupLayoutDescriptor::new(
            "ssgi_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // @binding(0) preprocessed_depth
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    // @binding(1) normals
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    // @binding(2) hilbert_index_lut
                    texture_2d(TextureSampleType::Uint),
                    // @binding(3) ssgi_output (rgba16float)
                    texture_storage_2d(SSGI_OUTPUT_FORMAT, StorageTextureAccess::WriteOnly),
                    // @binding(4) depth_differences
                    texture_storage_2d(TextureFormat::R32Uint, StorageTextureAccess::WriteOnly),
                    // @binding(5) globals
                    uniform_buffer::<GlobalsUniform>(false),
                    // @binding(6) thickness
                    uniform_buffer::<f32>(false),
                    // @binding(7) deferred_prepass (u32 GBuffer)
                    texture_2d(TextureSampleType::Uint),
                ),
            ),
        );

        // Spatial denoise: reads noisy SSGI (rgba16float via texture_2d) + depth differences;
        // writes denoised SSGI (rgba16float).
        let spatial_denoise_bind_group_layout = BindGroupLayoutDescriptor::new(
            "ssgi_spatial_denoise_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // @binding(0) ssgi_noisy (read as texture_2d<f32>)
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    // @binding(1) depth_differences
                    texture_2d(TextureSampleType::Uint),
                    // @binding(2) ssgi_denoised (write as rgba16float storage)
                    texture_storage_2d(SSGI_OUTPUT_FORMAT, StorageTextureAccess::WriteOnly),
                ),
            ),
        );

        let mut depth_shader_defs = Vec::new();
        if depth_format == TextureFormat::R16Float {
            depth_shader_defs.push("USE_R16FLOAT".into());
        }

        let preprocess_depth_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("ssgi_preprocess_depth_pipeline".into()),
                layout: vec![
                    preprocess_depth_bind_group_layout.clone(),
                    common_bind_group_layout.clone(),
                ],
                shader: load_embedded_asset!(world, "preprocess_depth.wgsl"),
                shader_defs: depth_shader_defs.clone(),
                ..default()
            });

        let spatial_denoise_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("ssgi_spatial_denoise_pipeline".into()),
                layout: vec![
                    spatial_denoise_bind_group_layout.clone(),
                    common_bind_group_layout.clone(),
                ],
                shader: load_embedded_asset!(world, "spatial_denoise.wgsl"),
                // The spatial denoiser no longer uses USE_R16FLOAT since it works
                // with Rgba16Float, but it doesn't hurt to pass no shader defs.
                shader_defs: Vec::new(),
                ..default()
            });

        Self {
            preprocess_depth_pipeline,
            spatial_denoise_pipeline,

            common_bind_group_layout,
            preprocess_depth_bind_group_layout,
            ssgi_bind_group_layout,
            spatial_denoise_bind_group_layout,

            hilbert_index_lut,
            point_clamp_sampler,
            linear_clamp_sampler,

            shader: load_embedded_asset!(world, "ssgi.wgsl"),
            depth_format,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
struct SsgiPipelineKey {
    quality_level: ScreenSpaceGlobalIlluminationQualityLevel,
    temporal_jitter: bool,
}

impl SpecializedComputePipeline for SsgiPipelines {
    type Key = SsgiPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let (slice_count, samples_per_slice_side) = key.quality_level.sample_counts();

        let mut shader_defs = vec![
            ShaderDefVal::Int("SLICE_COUNT".to_string(), slice_count as i32),
            ShaderDefVal::Int(
                "SAMPLES_PER_SLICE_SIDE".to_string(),
                samples_per_slice_side as i32,
            ),
        ];

        if key.temporal_jitter {
            shader_defs.push("TEMPORAL_JITTER".into());
        }

        ComputePipelineDescriptor {
            label: Some("ssgi_pipeline".into()),
            layout: vec![
                self.ssgi_bind_group_layout.clone(),
                self.common_bind_group_layout.clone(),
            ],
            shader: self.shader.clone(),
            shader_defs,
            ..default()
        }
    }
}

fn extract_ssgi_settings(
    mut commands: Commands,
    cameras: Extract<
        Query<
            (RenderEntity, &Camera, &ScreenSpaceGlobalIllumination, &Msaa),
            (
                With<Camera3d>,
                With<DepthPrepass>,
                With<NormalPrepass>,
                With<DeferredPrepass>,
            ),
        >,
    >,
) {
    for (entity, camera, ssgi_settings, msaa) in &cameras {
        if *msaa != Msaa::Off {
            error!(
                "SSGI is being used which requires Msaa::Off, but Msaa is currently set to Msaa::{:?}",
                *msaa
            );
            return;
        }
        let mut entity_commands = commands
            .get_entity(entity)
            .expect("SSGI entity wasn't synced.");
        if camera.is_active {
            entity_commands.insert(ssgi_settings.clone());
        } else {
            entity_commands.remove::<ScreenSpaceGlobalIllumination>();
        }
    }
}

#[derive(Component)]
pub struct ScreenSpaceGlobalIlluminationResources {
    preprocessed_depth_texture: CachedTexture,
    ssgi_noisy_texture: CachedTexture,
    /// The spatially-denoised SSGI texture.
    /// R = ambient occlusion, GBA = indirect light color.
    pub screen_space_ambient_occlusion_texture: CachedTexture,
    depth_differences_texture: CachedTexture,
    thickness_buffer: Buffer,
}

fn prepare_ssgi_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    pipelines: Res<SsgiPipelines>,
    views: Query<(Entity, &ExtractedCamera, &ScreenSpaceGlobalIllumination)>,
) {
    for (entity, camera, ssgi_settings) in &views {
        let Some(physical_viewport_size) = camera.physical_viewport_size else {
            continue;
        };
        let size = physical_viewport_size.to_extents();

        let preprocessed_depth_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("ssgi_preprocessed_depth_texture"),
                size,
                mip_level_count: 5,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: pipelines.depth_format,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        let ssgi_noisy_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("ssgi_noisy_texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SSGI_OUTPUT_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        let ssgi_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("ssgi_texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SSGI_OUTPUT_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        let depth_differences_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("ssgi_depth_differences_texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R32Uint,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        let thickness_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("ssgi_thickness_buffer"),
            contents: &ssgi_settings.constant_object_thickness.to_le_bytes(),
            usage: BufferUsages::UNIFORM,
        });

        commands
            .entity(entity)
            .insert(ScreenSpaceGlobalIlluminationResources {
                preprocessed_depth_texture,
                ssgi_noisy_texture,
                screen_space_ambient_occlusion_texture: ssgi_texture,
                depth_differences_texture,
                thickness_buffer,
            });
    }
}

#[derive(Component)]
struct SsgiPipelineId(CachedComputePipelineId);

fn prepare_ssgi_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedComputePipelines<SsgiPipelines>>,
    pipeline: Res<SsgiPipelines>,
    views: Query<(Entity, &ScreenSpaceGlobalIllumination, Has<TemporalJitter>)>,
) {
    for (entity, ssgi_settings, temporal_jitter) in &views {
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            SsgiPipelineKey {
                quality_level: ssgi_settings.quality_level,
                temporal_jitter,
            },
        );

        commands.entity(entity).insert(SsgiPipelineId(pipeline_id));
    }
}

#[derive(Component)]
struct SsgiBindGroups {
    common_bind_group: BindGroup,
    preprocess_depth_bind_group: BindGroup,
    ssgi_bind_group: BindGroup,
    spatial_denoise_bind_group: BindGroup,
}

fn prepare_ssgi_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipelines: Res<SsgiPipelines>,
    view_uniforms: Res<ViewUniforms>,
    global_uniforms: Res<GlobalsBuffer>,
    pipeline_cache: Res<PipelineCache>,
    views: Query<(
        Entity,
        &ScreenSpaceGlobalIlluminationResources,
        &ViewPrepassTextures,
    )>,
) {
    let (Some(view_uniforms), Some(globals_uniforms)) = (
        view_uniforms.uniforms.binding(),
        global_uniforms.buffer.binding(),
    ) else {
        return;
    };

    for (entity, ssgi_resources, prepass_textures) in &views {
        // We need the deferred prepass texture for GI sampling.
        let Some(deferred_view) = prepass_textures.deferred_view() else {
            continue;
        };

        let common_bind_group = render_device.create_bind_group(
            "ssgi_common_bind_group",
            &pipeline_cache.get_bind_group_layout(&pipelines.common_bind_group_layout),
            &BindGroupEntries::sequential((
                &pipelines.point_clamp_sampler,
                &pipelines.linear_clamp_sampler,
                view_uniforms.clone(),
            )),
        );

        let create_depth_view = |mip_level| {
            ssgi_resources
                .preprocessed_depth_texture
                .texture
                .create_view(&TextureViewDescriptor {
                    label: Some("ssgi_preprocessed_depth_texture_mip_view"),
                    base_mip_level: mip_level,
                    format: Some(pipelines.depth_format),
                    dimension: Some(TextureViewDimension::D2),
                    mip_level_count: Some(1),
                    ..default()
                })
        };

        let preprocess_depth_bind_group = render_device.create_bind_group(
            "ssgi_preprocess_depth_bind_group",
            &pipeline_cache.get_bind_group_layout(&pipelines.preprocess_depth_bind_group_layout),
            &BindGroupEntries::sequential((
                prepass_textures.depth_view().unwrap(),
                &create_depth_view(0),
                &create_depth_view(1),
                &create_depth_view(2),
                &create_depth_view(3),
                &create_depth_view(4),
            )),
        );

        let ssgi_bind_group = render_device.create_bind_group(
            "ssgi_bind_group",
            &pipeline_cache.get_bind_group_layout(&pipelines.ssgi_bind_group_layout),
            &BindGroupEntries::sequential((
                &ssgi_resources.preprocessed_depth_texture.default_view,
                prepass_textures.normal_view().unwrap(),
                &pipelines.hilbert_index_lut,
                &ssgi_resources.ssgi_noisy_texture.default_view,
                &ssgi_resources.depth_differences_texture.default_view,
                globals_uniforms.clone(),
                ssgi_resources.thickness_buffer.as_entire_binding(),
                deferred_view,
            )),
        );

        let spatial_denoise_bind_group = render_device.create_bind_group(
            "ssgi_spatial_denoise_bind_group",
            &pipeline_cache.get_bind_group_layout(&pipelines.spatial_denoise_bind_group_layout),
            &BindGroupEntries::sequential((
                &ssgi_resources.ssgi_noisy_texture.default_view,
                &ssgi_resources.depth_differences_texture.default_view,
                &ssgi_resources
                    .screen_space_ambient_occlusion_texture
                    .default_view,
            )),
        );

        commands.entity(entity).insert(SsgiBindGroups {
            common_bind_group,
            preprocess_depth_bind_group,
            ssgi_bind_group,
            spatial_denoise_bind_group,
        });
    }
}

fn generate_hilbert_index_lut() -> [[u16; 64]; 64] {
    use core::array::from_fn;
    from_fn(|x| from_fn(|y| hilbert_index(x as u16, y as u16)))
}

// https://www.shadertoy.com/view/3tB3z3
const HILBERT_WIDTH: u16 = 64;
fn hilbert_index(mut x: u16, mut y: u16) -> u16 {
    let mut index = 0;

    let mut level: u16 = HILBERT_WIDTH / 2;
    while level > 0 {
        let region_x = (x & level > 0) as u16;
        let region_y = (y & level > 0) as u16;
        index += level * level * ((3 * region_x) ^ region_y);

        if region_y == 0 {
            if region_x == 1 {
                x = HILBERT_WIDTH - 1 - x;
                y = HILBERT_WIDTH - 1 - y;
            }

            mem::swap(&mut x, &mut y);
        }

        level /= 2;
    }

    index
}

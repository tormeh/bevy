//! Demonstrates screen space ambient occlusion (SSAO) in deferred rendering.

use std::fmt;

use bevy::{
    anti_alias::taa::TemporalAntiAliasing,
    camera::Hdr,
    color::palettes::css::{BLACK, WHITE},
    core_pipeline::prepass::DeferredPrepass,
    image::{
        ImageAddressMode, ImageFilterMode, ImageLoaderSettings, ImageSampler,
        ImageSamplerDescriptor,
    },
    input::mouse::MouseWheel,
    light::Skybox,
    math::{vec3, vec4},
    pbr::{
        DefaultOpaqueRendererMethod, ExtendedMaterial, MaterialExtension,
        ScreenSpaceAmbientOcclusion, ScreenSpaceAmbientOcclusionQualityLevel,
    },
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
};

#[path = "../helpers/widgets.rs"]
mod widgets;

use widgets::{
    handle_ui_interactions, main_ui_node, option_buttons, update_ui_radio_button,
    update_ui_radio_button_text, RadioButton, RadioButtonText, WidgetClickEvent, WidgetClickSender,
    BUTTON_BORDER, BUTTON_BORDER_COLOR, BUTTON_BORDER_RADIUS_SIZE, BUTTON_PADDING,
};

/// This example uses a shader source file from the assets subdirectory
const SHADER_ASSET_PATH: &str = "shaders/water_material.wgsl";

// The speed of camera movement.
const CAMERA_KEYBOARD_ZOOM_SPEED: f32 = 0.1;
const CAMERA_KEYBOARD_ORBIT_SPEED: f32 = 0.02;
const CAMERA_MOUSE_WHEEL_ZOOM_SPEED: f32 = 0.25;

// We clamp camera distances to this range.
const CAMERA_ZOOM_RANGE: core::ops::Range<f32> = 2.0..12.0;

/// A custom [`ExtendedMaterial`] that creates animated water ripples.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct Water {
    /// The normal map image.
    ///
    /// Note that, like all normal maps, this must not be loaded as sRGB.
    #[texture(100)]
    #[sampler(101)]
    normals: Handle<Image>,

    // Parameters to the water shader.
    #[uniform(102)]
    settings: WaterSettings,
}

/// Parameters to the water shader.
#[derive(ShaderType, Debug, Clone)]
struct WaterSettings {
    /// How much to displace each octave each frame, in the u and v directions.
    /// Two octaves are packed into each `vec4`.
    octave_vectors: [Vec4; 2],
    /// How wide the waves are in each octave.
    octave_scales: Vec4,
    /// How high the waves are in each octave.
    octave_strengths: Vec4,
}

/// The current settings that the user has chosen.
#[derive(Resource)]
struct AppSettings {
    /// Whether screen space ambient occlusion is on.
    ssao_on: bool,
    /// The SSAO quality level.
    ssao_quality: ScreenSpaceAmbientOcclusionQualityLevel,
    /// The constant object thickness for SSAO.
    ssao_constant_object_thickness: f32,
    /// Which model is being displayed.
    displayed_model: DisplayedModel,
    /// Which base is being displayed.
    displayed_base: DisplayedBase,
}

/// Which model is being displayed.
#[derive(Default, PartialEq, Copy, Clone)]
enum DisplayedModel {
    /// The cube is being displayed.
    #[default]
    Cube,
    /// The flight helmet is being displayed.
    FlightHelmet,
    /// The capsules are being displayed.
    Capsules,
}

/// Which base is being displayed.
#[derive(Default, PartialEq, Copy, Clone)]
enum DisplayedBase {
    /// The water base is being displayed.
    #[default]
    Water,
    /// A slightly rough metallic base is being displayed.
    Metallic,
    /// A very rough non-metallic base is being displayed.
    RedPlane,
}

impl fmt::Display for DisplayedModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            DisplayedModel::Cube => "Cube",
            DisplayedModel::FlightHelmet => "Flight Helmet",
            DisplayedModel::Capsules => "Capsules",
        };
        write!(f, "{}", name)
    }
}

impl fmt::Display for DisplayedBase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            DisplayedBase::Water => "Water",
            DisplayedBase::Metallic => "Metallic",
            DisplayedBase::RedPlane => "Red Plane",
        };
        write!(f, "{}", name)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ExampleSetting {
    Ssao(bool),
    SsaoQuality(ScreenSpaceAmbientOcclusionQualityLevel),
    ThicknessIncrease,
    ThicknessDecrease,
    Model(DisplayedModel),
    Base(DisplayedBase),
}

/// A marker component for the single cube model.
#[derive(Component)]
struct CubeModel;

/// A marker component for the flight helmet model.
#[derive(Component)]
struct FlightHelmetModel;

/// A marker component for the row of capsules model.
#[derive(Component)]
struct CapsuleModel;

/// A marker component for the row of capsules parent.
#[derive(Component)]
struct CapsulesParent;

/// A marker component for the metallic base.
#[derive(Component)]
struct MetallicBaseModel;

/// A marker component for the non-metallic base.
#[derive(Component)]
struct RedPlaneBaseModel;

/// A marker component for the water model.
#[derive(Component)]
struct WaterModel;

/// A marker component for the text that displays the thickness value.
#[derive(Component)]
struct ThicknessValueText;

#[derive(bevy::ecs::system::SystemParam)]
struct ModelQueries<'w, 's> {
    cube_models: Query<'w, 's, Entity, With<CubeModel>>,
    flight_helmet_models: Query<'w, 's, Entity, With<FlightHelmetModel>>,
    capsule_models: Query<'w, 's, Entity, Or<(With<CapsuleModel>, With<CapsulesParent>)>>,
    metallic_base_models: Query<'w, 's, Entity, With<MetallicBaseModel>>,
    non_metallic_base_models: Query<'w, 's, Entity, With<RedPlaneBaseModel>>,
    water_models: Query<'w, 's, Entity, With<WaterModel>>,
}

fn main() {
    // Enable deferred rendering, which is necessary for screen-space
    // ambient occlusion at this time. Disable multisampled antialiasing, as
    // deferred rendering doesn't support that.
    App::new()
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .init_resource::<AppSettings>()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bevy Screen Space Ambient Occlusion Example".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(MaterialPlugin::<ExtendedMaterial<StandardMaterial, Water>>::default())
        .add_message::<WidgetClickEvent<ExampleSetting>>()
        .add_systems(Startup, setup)
        .add_systems(Update, rotate_model)
        .add_systems(Update, move_camera)
        .add_systems(Update, adjust_app_settings)
        .add_systems(Update, handle_ui_interactions::<ExampleSetting>)
        .run();
}

// Set up the scene.
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
    mut water_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, Water>>>,
    asset_server: Res<AssetServer>,
    app_settings: Res<AppSettings>,
) {
    spawn_cube(
        &mut commands,
        &asset_server,
        &mut meshes,
        &mut standard_materials,
    );
    spawn_flight_helmet(&mut commands, &asset_server);
    spawn_capsules(&mut commands, &mut meshes, &mut standard_materials);
    spawn_metallic_base(&mut commands, &mut meshes, &mut standard_materials);
    spawn_non_metallic_base(&mut commands, &mut meshes, &mut standard_materials);
    spawn_water(
        &mut commands,
        &asset_server,
        &mut meshes,
        &mut water_materials,
    );
    spawn_camera(&mut commands, &asset_server, &app_settings);
    spawn_buttons(&mut commands, &app_settings);
}

// Spawns the rotating cube.
fn spawn_cube(
    commands: &mut Commands,
    asset_server: &AssetServer,
    meshes: &mut Assets<Mesh>,
    standard_materials: &mut Assets<StandardMaterial>,
) {
    commands
        .spawn((
            Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
            MeshMaterial3d(standard_materials.add(StandardMaterial {
                base_color: Color::from(WHITE),
                base_color_texture: Some(asset_server.load("branding/icon.png")),
                ..default()
            })),
            Transform::from_xyz(0.0, 0.5, 0.0),
        ))
        .insert(CubeModel);
}

// Spawns the flight helmet.
fn spawn_flight_helmet(commands: &mut Commands, asset_server: &AssetServer) {
    commands.spawn((
        SceneRoot(
            asset_server
                .load(GltfAssetLabel::Scene(0).from_asset("models/FlightHelmet/FlightHelmet.gltf")),
        ),
        Transform::from_scale(Vec3::splat(2.5)),
        FlightHelmetModel,
        Visibility::Hidden,
    ));
}

// Spawns the row of capsules.
fn spawn_capsules(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    standard_materials: &mut Assets<StandardMaterial>,
) {
    let capsule_mesh = meshes.add(Capsule3d::new(0.4, 0.5));
    let parent = commands
        .spawn((
            Transform::from_xyz(0.0, 0.5, 0.0),
            Visibility::Hidden,
            CapsulesParent,
        ))
        .id();

    for i in 0..5 {
        let roughness = i as f32 * 0.25;
        let child = commands
            .spawn((
                Mesh3d(capsule_mesh.clone()),
                MeshMaterial3d(standard_materials.add(StandardMaterial {
                    base_color: Color::BLACK,
                    perceptual_roughness: roughness.max(0.08),
                    ..default()
                })),
                Transform::from_xyz(i as f32 * 1.1 - (1.1 * 2.0), 0.5, 0.0),
                CapsuleModel,
            ))
            .id();
        commands.entity(parent).add_child(child);
    }
}

// Spawns the metallic base.
fn spawn_metallic_base(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    standard_materials: &mut Assets<StandardMaterial>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(1.0)))),
        MeshMaterial3d(standard_materials.add(StandardMaterial {
            base_color: Color::from(bevy::color::palettes::css::DARK_GRAY),
            metallic: 1.0,
            perceptual_roughness: 0.3,
            ..default()
        })),
        Transform::from_scale(Vec3::splat(100.0)),
        MetallicBaseModel,
        Visibility::Hidden,
    ));
}

// Spawns the non-metallic base.
fn spawn_non_metallic_base(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    standard_materials: &mut Assets<StandardMaterial>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(1.0)))),
        MeshMaterial3d(standard_materials.add(StandardMaterial {
            base_color: Color::from(bevy::color::palettes::css::RED),
            metallic: 0.0,
            perceptual_roughness: 0.2,
            ..default()
        })),
        Transform::from_scale(Vec3::splat(100.0)),
        RedPlaneBaseModel,
        Visibility::Hidden,
    ));
}

// Spawns the water plane.
fn spawn_water(
    commands: &mut Commands,
    asset_server: &AssetServer,
    meshes: &mut Assets<Mesh>,
    water_materials: &mut Assets<ExtendedMaterial<StandardMaterial, Water>>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(1.0)))),
        MeshMaterial3d(water_materials.add(ExtendedMaterial {
            base: StandardMaterial {
                base_color: BLACK.into(),
                perceptual_roughness: 0.09,
                ..default()
            },
            extension: Water {
                normals: asset_server.load_with_settings::<Image, ImageLoaderSettings>(
                    "textures/water_normals.png",
                    |settings| {
                        settings.is_srgb = false;
                        settings.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                            address_mode_u: ImageAddressMode::Repeat,
                            address_mode_v: ImageAddressMode::Repeat,
                            mag_filter: ImageFilterMode::Linear,
                            min_filter: ImageFilterMode::Linear,
                            ..default()
                        });
                    },
                ),
                // These water settings are just random values to create some
                // variety.
                settings: WaterSettings {
                    octave_vectors: [
                        vec4(0.080, 0.059, 0.073, -0.062),
                        vec4(0.153, 0.138, -0.149, -0.195),
                    ],
                    octave_scales: vec4(1.0, 2.1, 7.9, 14.9) * 5.0,
                    octave_strengths: vec4(0.16, 0.18, 0.093, 0.044),
                },
            },
        })),
        Transform::from_scale(Vec3::splat(100.0)),
        WaterModel,
    ));
}

// Spawns the camera.
fn spawn_camera(commands: &mut Commands, asset_server: &AssetServer, app_settings: &AppSettings) {
    // Create the camera. Add an environment map and skybox so the scene has
    // interesting lighting. Enable deferred rendering by adding depth and
    // deferred prepasses. Turn on TAA to make the scene look a little nicer.
    // Finally, add screen space ambient occlusion.
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(vec3(-1.25, 2.25, 4.5)).looking_at(Vec3::ZERO, Vec3::Y),
        Hdr,
        Msaa::Off,
        TemporalAntiAliasing::default(),
        DeferredPrepass,
        ScreenSpaceAmbientOcclusion {
            quality_level: app_settings.ssao_quality,
            constant_object_thickness: app_settings.ssao_constant_object_thickness,
        },
        EnvironmentMapLight {
            diffuse_map: asset_server.load("environment_maps/pisa_diffuse_rgb9e5_zstd.ktx2"),
            specular_map: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            intensity: 5000.0,
            ..default()
        },
        Skybox {
            image: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            brightness: 5000.0,
            ..default()
        },
    ));
}

fn spawn_buttons(commands: &mut Commands, app_settings: &AppSettings) {
    commands.spawn(main_ui_node()).with_children(|parent| {
        parent.spawn(option_buttons(
            "SSAO",
            &[
                (ExampleSetting::Ssao(true), "On"),
                (ExampleSetting::Ssao(false), "Off"),
            ],
        ));

        parent.spawn(option_buttons(
            "Quality",
            &[
                (
                    ExampleSetting::SsaoQuality(ScreenSpaceAmbientOcclusionQualityLevel::Low),
                    "Low",
                ),
                (
                    ExampleSetting::SsaoQuality(ScreenSpaceAmbientOcclusionQualityLevel::Medium),
                    "Medium",
                ),
                (
                    ExampleSetting::SsaoQuality(ScreenSpaceAmbientOcclusionQualityLevel::High),
                    "High",
                ),
                (
                    ExampleSetting::SsaoQuality(ScreenSpaceAmbientOcclusionQualityLevel::Ultra),
                    "Ultra",
                ),
            ],
        ));

        parent.spawn(thickness_row(app_settings.ssao_constant_object_thickness));

        parent.spawn(option_buttons(
            "Model",
            &[
                (ExampleSetting::Model(DisplayedModel::Cube), "Cube"),
                (
                    ExampleSetting::Model(DisplayedModel::FlightHelmet),
                    "Flight Helmet",
                ),
                (ExampleSetting::Model(DisplayedModel::Capsules), "Capsules"),
            ],
        ));

        parent.spawn(option_buttons(
            "Base",
            &[
                (ExampleSetting::Base(DisplayedBase::Water), "Water"),
                (ExampleSetting::Base(DisplayedBase::Metallic), "Metallic"),
                (ExampleSetting::Base(DisplayedBase::RedPlane), "Red Plane"),
            ],
        ));
    });
}

fn thickness_row(value: f32) -> impl Bundle {
    (
        Node {
            align_items: AlignItems::Center,
            ..default()
        },
        Children::spawn((
            Spawn((
                widgets::ui_text("Thickness", Color::WHITE),
                Node {
                    width: px(150),
                    ..default()
                },
            )),
            Spawn(thickness_controls(value)),
        )),
    )
}

fn thickness_controls(value: f32) -> impl Bundle {
    (
        Node {
            align_items: AlignItems::Center,
            ..default()
        },
        Children::spawn((
            Spawn(adjustment_button(
                ExampleSetting::ThicknessDecrease,
                "<",
                Some(true),
            )),
            Spawn((
                Node {
                    width: px(60),
                    height: px(33),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    border: BUTTON_BORDER.with_left(px(0)).with_right(px(0)),
                    ..default()
                },
                BackgroundColor(Color::WHITE),
                BUTTON_BORDER_COLOR,
                ThicknessValueText,
                children![(widgets::ui_text(&format!("{:.4}", value), Color::BLACK))],
            )),
            Spawn(adjustment_button(
                ExampleSetting::ThicknessIncrease,
                ">",
                Some(false),
            )),
        )),
    )
}

fn adjustment_button(
    setting: ExampleSetting,
    label: &str,
    is_left_right: Option<bool>,
) -> impl Bundle {
    (
        Button,
        Node {
            height: px(33),
            border: if let Some(is_left) = is_left_right {
                if is_left {
                    BUTTON_BORDER.with_right(px(0))
                } else {
                    BUTTON_BORDER.with_left(px(0))
                }
            } else {
                BUTTON_BORDER
            },
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            padding: BUTTON_PADDING,
            border_radius: match is_left_right {
                Some(true) => BorderRadius::ZERO.with_left(BUTTON_BORDER_RADIUS_SIZE),
                Some(false) => BorderRadius::ZERO.with_right(BUTTON_BORDER_RADIUS_SIZE),
                None => BorderRadius::all(BUTTON_BORDER_RADIUS_SIZE),
            },
            ..default()
        },
        BUTTON_BORDER_COLOR,
        BackgroundColor(Color::BLACK),
        RadioButton,
        WidgetClickSender(setting),
        children![(widgets::ui_text(label, Color::WHITE), RadioButtonText)],
    )
}

fn rotate_model(
    mut query: Query<&mut Transform, Or<(With<CubeModel>, With<FlightHelmetModel>)>>,
    time: Res<Time>,
) {
    for mut transform in query.iter_mut() {
        // Models rotate on the Y axis.
        transform.rotation = Quat::from_rotation_y(time.elapsed_secs());
    }
}

// Processes input related to camera movement.
fn move_camera(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut mouse_wheel_reader: MessageReader<MouseWheel>,
    mut cameras: Query<&mut Transform, With<Camera>>,
) {
    let (mut distance_delta, mut theta_delta) = (0.0, 0.0);

    // Handle keyboard events.
    if keyboard_input.pressed(KeyCode::KeyW) {
        distance_delta -= CAMERA_KEYBOARD_ZOOM_SPEED;
    }
    if keyboard_input.pressed(KeyCode::KeyS) {
        distance_delta += CAMERA_KEYBOARD_ZOOM_SPEED;
    }
    if keyboard_input.pressed(KeyCode::KeyA) {
        theta_delta += CAMERA_KEYBOARD_ORBIT_SPEED;
    }
    if keyboard_input.pressed(KeyCode::KeyD) {
        theta_delta -= CAMERA_KEYBOARD_ORBIT_SPEED;
    }

    // Handle mouse events.
    for mouse_wheel in mouse_wheel_reader.read() {
        distance_delta -= mouse_wheel.y * CAMERA_MOUSE_WHEEL_ZOOM_SPEED;
    }

    // Update transforms.
    for mut camera_transform in cameras.iter_mut() {
        let local_z = camera_transform.local_z().as_vec3().normalize_or_zero();
        if distance_delta != 0.0 {
            camera_transform.translation = (camera_transform.translation.length() + distance_delta)
                .clamp(CAMERA_ZOOM_RANGE.start, CAMERA_ZOOM_RANGE.end)
                * local_z;
        }
        if theta_delta != 0.0 {
            camera_transform
                .translate_around(Vec3::ZERO, Quat::from_axis_angle(Vec3::Y, theta_delta));
            camera_transform.look_at(Vec3::ZERO, Vec3::Y);
        }
    }
}

// Adjusts app settings per user input.
fn adjust_app_settings(
    mut commands: Commands,
    mut app_settings: ResMut<AppSettings>,
    mut cameras: Query<Entity, With<Camera>>,
    mut visibilities: Query<&mut Visibility>,
    model_queries: ModelQueries,
    mut widget_click_events: MessageReader<WidgetClickEvent<ExampleSetting>>,
    mut background_colors: Query<&mut BackgroundColor>,
    radio_buttons: Query<
        (
            Entity,
            Has<BackgroundColor>,
            Has<Text>,
            &WidgetClickSender<ExampleSetting>,
        ),
        Or<(With<RadioButton>, With<RadioButtonText>)>,
    >,
    thickness_value_text: Query<Entity, With<ThicknessValueText>>,
    text_children: Query<&Children>,
    mut writer: TextUiWriter,
    text_query: Query<Entity, With<Text>>,
) {
    let mut any_changes = false;

    for event in widget_click_events.read() {
        any_changes = true;
        match **event {
            ExampleSetting::Ssao(on) => app_settings.ssao_on = on,
            ExampleSetting::SsaoQuality(quality) => app_settings.ssao_quality = quality,
            ExampleSetting::ThicknessIncrease => {
                app_settings.ssao_constant_object_thickness =
                    (app_settings.ssao_constant_object_thickness * 2.0).min(4.0);
            }
            ExampleSetting::ThicknessDecrease => {
                app_settings.ssao_constant_object_thickness =
                    (app_settings.ssao_constant_object_thickness * 0.5).max(0.0625);
            }
            ExampleSetting::Model(model) => app_settings.displayed_model = model,
            ExampleSetting::Base(base) => app_settings.displayed_base = base,
        }
    }

    if !any_changes {
        return;
    }

    // Update SSAO settings.
    for camera in cameras.iter_mut() {
        if app_settings.ssao_on {
            commands.entity(camera).insert(ScreenSpaceAmbientOcclusion {
                quality_level: app_settings.ssao_quality,
                constant_object_thickness: app_settings.ssao_constant_object_thickness,
            });
        } else {
            commands
                .entity(camera)
                .remove::<ScreenSpaceAmbientOcclusion>();
        }
    }

    // Set model visibility.
    for entity in model_queries.cube_models.iter() {
        if let Ok(mut visibility) = visibilities.get_mut(entity) {
            *visibility = if app_settings.displayed_model == DisplayedModel::Cube {
                Visibility::Visible
            } else {
                Visibility::Hidden
            };
        }
    }
    for entity in model_queries.flight_helmet_models.iter() {
        if let Ok(mut visibility) = visibilities.get_mut(entity) {
            *visibility = if app_settings.displayed_model == DisplayedModel::FlightHelmet {
                Visibility::Visible
            } else {
                Visibility::Hidden
            };
        }
    }
    for entity in model_queries.capsule_models.iter() {
        if let Ok(mut visibility) = visibilities.get_mut(entity) {
            *visibility = if app_settings.displayed_model == DisplayedModel::Capsules {
                Visibility::Visible
            } else {
                Visibility::Hidden
            };
        }
    }
    for entity in model_queries.metallic_base_models.iter() {
        if let Ok(mut visibility) = visibilities.get_mut(entity) {
            *visibility = if app_settings.displayed_base == DisplayedBase::Metallic {
                Visibility::Visible
            } else {
                Visibility::Hidden
            };
        }
    }
    for entity in model_queries.non_metallic_base_models.iter() {
        if let Ok(mut visibility) = visibilities.get_mut(entity) {
            *visibility = if app_settings.displayed_base == DisplayedBase::RedPlane {
                Visibility::Visible
            } else {
                Visibility::Hidden
            };
        }
    }
    for entity in model_queries.water_models.iter() {
        if let Ok(mut visibility) = visibilities.get_mut(entity) {
            *visibility = if app_settings.displayed_base == DisplayedBase::Water {
                Visibility::Visible
            } else {
                Visibility::Hidden
            };
        }
    }

    // Update radio buttons.
    for (entity, has_background, has_text, sender) in radio_buttons.iter() {
        let selected = match **sender {
            ExampleSetting::Ssao(on) => app_settings.ssao_on == on,
            ExampleSetting::SsaoQuality(quality) => app_settings.ssao_quality == quality,
            ExampleSetting::Model(model) => app_settings.displayed_model == model,
            ExampleSetting::Base(base) => app_settings.displayed_base == base,
            _ => {
                if has_background
                    && let Ok(mut background_color) = background_colors.get_mut(entity)
                {
                    *background_color = BackgroundColor(Color::BLACK);
                }
                if has_text {
                    update_ui_radio_button_text(entity, &mut writer, false);
                }
                continue;
            }
        };

        if has_background && let Ok(mut background_color) = background_colors.get_mut(entity) {
            update_ui_radio_button(&mut background_color, selected);
        }
        if has_text {
            update_ui_radio_button_text(entity, &mut writer, selected);
        }
    }

    // Update thickness value text.
    for parent in thickness_value_text.iter() {
        if let Ok(children) = text_children.get(parent) {
            for child in children.iter() {
                if text_query.get(child).is_ok() {
                    *writer.text(child, 0) =
                        format!("{:.4}", app_settings.ssao_constant_object_thickness);
                    writer.for_each_color(child, |mut color| {
                        color.0 = Color::BLACK;
                    });
                }
            }
        }
    }
}

impl MaterialExtension for Water {
    fn deferred_fragment_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            ssao_on: true,
            ssao_quality: ScreenSpaceAmbientOcclusionQualityLevel::Medium,
            ssao_constant_object_thickness: 0.25,
            displayed_model: default(),
            displayed_base: default(),
        }
    }
}

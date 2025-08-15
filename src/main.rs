use bevy::prelude::*;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bytemuck::{Pod, Zeroable};

mod camera;
use camera::{pan_orbit_camera, PanOrbitCamera};

/// Physical constants (SI)
const EPS0: f32 = 8.854_187e-12; // Permittivity of free space
const MU0: f32 = 4.0 * std::f32::consts::PI * 1.0e-7; // Permeability of free space

/// Speed of light in vacuum
fn c0() -> f32 {
    1.0 / (MU0 * EPS0).sqrt()
}

/// Simulation grid parameters
const NX: usize = 300; // Number of grid points in x
const NY: usize = 200; // Number of grid points in y
const DX: f32 = 1.0; // Grid spacing in x
const DY: f32 = 1.0; // Grid spacing in y
const SUBSTEPS: usize = 4; // Number of simulation steps per frame

/// Courant-Friedrichs-Lewy (CFL) number for stability
const CFL: f32 = 0.5;

/// Damping border parameters for absorbing boundary conditions
const DAMP_BORDER: usize = 8;
const DAMP_STRENGTH: f32 = 0.015;

/// Source parameters
const SRC_AMP: f32 = 1.0; // Amplitude of the source
const SRC_STD_STEPS: f32 = 12.0; // Standard deviation of the Gaussian source pulse

/// The `Grid` resource holds the state of the FDTD simulation.
#[derive(Resource)]
struct Grid {
    nx: usize,
    ny: usize,
    dx: f32,
    dy: f32,
    dt: f32,
    ez: Vec<f32>, // z-component of the electric field
    hx: Vec<f32>, // x-component of the magnetic field
    hy: Vec<f32>, // y-component of the magnetic field
    viz_scale: f32, // Scaling factor for visualization
    step: u64,      // Simulation step counter
}

impl Grid {
    fn new(nx: usize, ny: usize, dx: f32, dy: f32) -> Self {
        // The time step `dt` is chosen to satisfy the CFL stability condition.
        let dt = CFL / (c0() * (1.0 / dx.powi(2) + 1.0 / dy.powi(2)).sqrt());
        Self {
            nx,
            ny,
            dx,
            dy,
            dt,
            ez: vec![0.0; nx * ny],
            hx: vec![0.0; nx * ny],
            hy: vec![0.0; nx * ny],
            viz_scale: 1.5,
            step: 0,
        }
    }

    /// Helper function to get the 1D index from 2D grid coordinates.
    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }
}

/// A simple RGBA color struct for the texture.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Rgba8 {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

/// The `FieldTexture` resource holds the handle to the image used for visualization.
#[derive(Resource)]
struct FieldTexture(Handle<Image>);

/// Enum to select which field to visualize.
#[derive(Resource, Default, Debug, PartialEq, Eq, Hash, States, Clone, Copy)]
enum VizField {
    #[default]
    Ez,
    Hx,
    Hy,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()))
        .insert_resource(Grid::new(NX, NY, DX, DY))
        .init_resource::<VizField>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                (fdtd_step_system, upload_texture_system).chain(),
                pan_orbit_camera,
                toggle_viz_field,
            ),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let mut image = Image::new_fill(
        Extent3d {
            width: NX as u32,
            height: NY as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0u8; 4],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_DST;

    let handle = images.add(image.clone());
    commands.insert_resource(FieldTexture(handle.clone()));

    // Spawn a plane to display the visualization texture.
    commands.spawn(PbrBundle {
        mesh: meshes.add(shape::Plane::from_size(5.0).into()),
        material: materials.add(StandardMaterial {
            base_color_texture: Some(handle),
            unlit: true,
            ..default()
        }),
        ..default()
    });

    // Spawn the camera
    let translation = Vec3::new(0.0, 0.0, 5.0);
    let radius = translation.length();
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(translation).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        PanOrbitCamera { radius, ..default() },
    ));

    let g = Grid::new(NX, NY, DX, DY);
    info!("Δt = {:.3e} s, c0Δt/Δx ≈ {:.3}", g.dt, c0() * g.dt / DX);
    info!("Press E to visualize Ez field");
    info!("Press H to visualize Hx field");
    info!("Press Y to visualize Hy field");
}

/// This system runs the FDTD simulation for a few substeps.
fn fdtd_step_system(mut grid: ResMut<Grid>) {
    for _ in 0..SUBSTEPS {
        update_h(&mut grid);
        inject_source(grid);
        update_e(&mut grid);
        apply_simple_absorbing(&mut grid);
        grid.step += 1;
    }
}

/// Update the magnetic field (H) based on the curl of the electric field (E).
/// This is Faraday's Law: ∇ × E = -∂B/∂t
fn update_h(grid: &mut Grid) {
    let nx = grid.nx;
    let ny = grid.ny;
    let dt_over_mu_dx = grid.dt / (MU0 * grid.dx);
    let dt_over_mu_dy = grid.dt / (MU0 * grid.dy);

    for j in 0..ny - 1 {
        for i in 0..nx {
            let idx = grid.idx(i, j);
            let idx_next = grid.idx(i, j + 1);
            let d_ez_dy = grid.ez[idx_next] - grid.ez[idx];
            grid.hx[idx] -= dt_over_mu_dy * d_ez_dy;
        }
    }
    for j in 0..ny {
        for i in 0..nx - 1 {
            let idx = grid.idx(i, j);
            let idx_next = grid.idx(i + 1, j);
            let d_ez_dx = grid.ez[idx_next] - grid.ez[idx];
            grid.hy[idx] += dt_over_mu_dx * d_ez_dx;
        }
    }
}

/// Update the electric field (E) based on the curl of the magnetic field (H).
/// This is Ampere-Maxwell's Law: ∇ × H = J + ∂D/∂t
fn update_e(grid: &mut Grid) {
    let nx = grid.nx;
    let ny = grid.ny;
    let dt_over_eps_dx = grid.dt / (EPS0 * grid.dx);
    let dt_over_eps_dy = grid.dt / (EPS0 * grid.dy);

    for j in 1..ny {
        for i in 1..nx {
            let idx = grid.idx(i, j);
            let idx_xm = grid.idx(i - 1, j);
            let idx_ym = grid.idx(i, j - 1);
            let d_hy_dx = grid.hy[idx] - grid.hy[idx_xm];
            let d_hx_dy = grid.hx[idx] - grid.hx[idx_ym];
            grid.ez[idx] += dt_over_eps_dx * d_hy_dx - dt_over_eps_dy * d_hx_dy;
        }
    }

    // Set electric field to zero on the boundaries (Perfect Electric Conductor)
    for i in 0..nx {
        let top = grid.idx(i, 0);
        grid.ez[top] = 0.0;
        let bottom = grid.idx(i, ny - 1);
        grid.ez[bottom] = 0.0;
    }
    for j in 0..ny {
        let left = grid.idx(0, j);
        grid.ez[left] = 0.0;
        let right = grid.idx(nx - 1, j);
        grid.ez[right] = 0.0;
    }
}

/// Inject a source into the grid.
/// Here we use a Gaussian pulse to excite the Ez field.
fn inject_source(mut grid: ResMut<Grid>) {
    let i0 = grid.nx / 2;
    let j0 = grid.ny / 2;
    let t = grid.step as f32;
    let t0 = 4.0 * SRC_STD_STEPS;
    let gauss = (-((t - t0).powi(2)) / (2.0 * SRC_STD_STEPS.powi(2))).exp();
    let ez_kick = -grid.dt * (SRC_AMP * gauss) / EPS0;

    for dj in -1..=1 {
        for di in -1..=1 {
            let i = (i0 as isize + di) as usize;
            let j = (j0 as isize + dj) as usize;
            let idx = grid.idx(i, j);
            let w = if di == 0 && dj == 0 { 1.0 } else { 0.6 };
            grid.ez[idx] += w * ez_kick;
        }
    }
}

/// Apply a simple absorbing boundary condition to prevent reflections from the boundaries.
fn apply_simple_absorbing(grid: &mut Grid) {
    let nx = grid.nx;
    let ny = grid.ny;
    let mut wx = vec![1.0f32; nx];
    let mut wy = vec![1.0f32; ny];

    for i in 0..DAMP_BORDER.min(nx / 2) {
        let f = (i as f32) / (DAMP_BORDER as f32);
        let w = (1.0 - DAMP_STRENGTH * (1.0 - f)).max(0.0);
        wx[i] *= w;
        wx[nx - 1 - i] *= w;
    }
    for j in 0..DAMP_BORDER.min(ny / 2) {
        let f = (j as f32) / (DAMP_BORDER as f32);
        let w = (1.0 - DAMP_STRENGTH * (1.0 - f)).max(0.0);
        wy[j] *= w;
        wy[ny - 1 - j] *= w;
    }

    for j in 0..ny {
        for i in 0..nx {
            let idx = grid.idx(i, j);
            grid.ez[idx] *= wx[i] * wy[j];
        }
    }
}

/// This system uploads the selected field data to a texture for visualization.
fn upload_texture_system(
    grid: Res<Grid>,
    field_tex: Res<FieldTexture>,
    mut images: ResMut<Assets<Image>>,
    viz_field: Res<VizField>,
) {
    if let Some(img) = images.get_mut(&field_tex.0) {
        let mut pixels: Vec<Rgba8> = Vec::with_capacity(grid.nx * grid.ny);
        let field_to_viz = match *viz_field {
            VizField::Ez => &grid.ez,
            VizField::Hx => &grid.hx,
            VizField::Hy => &grid.hy,
        };

        for val in field_to_viz.iter() {
            let x = val * grid.viz_scale;
            let x = x.clamp(-1.0, 1.0);
            let r = ((x.max(0.0)) * 255.0) as u8;
            let b = ((-x.max(0.0)) * 255.0) as u8;
            let g = (255.0 - ((x.abs()) * 255.0 * 0.5)).clamp(0.0, 255.0) as u8;
            pixels.push(Rgba8 { r, g, b, a: 255 });
        }

        img.resize(Extent3d {
            width: grid.nx as u32,
            height: grid.ny as u32,
            depth_or_array_layers: 1,
        });
        img.data = Some(bytemuck::cast_slice(&pixels).to_vec());
    }
}

/// System to toggle which field is being visualized.
fn toggle_viz_field(mut viz_field: ResMut<VizField>, keyboard_input: Res<ButtonInput<KeyCode>>) {
    if keyboard_input.just_pressed(KeyCode::KeyE) {
        *viz_field = VizField::Ez;
        info!("Visualizing Ez field");
    }
    if keyboard_input.just_pressed(KeyCode::KeyH) {
        *viz_field = VizField::Hx;
        info!("Visualizing Hx field");
    }
    if keyboard_input.just_pressed(KeyCode::KeyY) {
        *viz_field = VizField::Hy;
        info!("Visualizing Hy field");
    }
}
// use bevy::prelude::*;
// use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
// use bytemuck::{Pod, Zeroable};

// /// Physical constants (SI)
// const EPS0: f32 = 8.854_187e-12;
// const MU0: f32 = 4.0 * std::f32::consts::PI * 1.0e-7;
// const C0: f32 = 1.0 / (MU0 * EPS0).sqrt();

// /// Simulation grid
// const NX: usize = 300;
// const NY: usize = 200;
// const DX: f32 = 1.0; // meters (arbitrary unit scale)
// const DY: f32 = 1.0;

// /// Courant stability factor (<= 1/sqrt(2) for 2D)
// const CFL: f32 = 0.5;

// /// Sub-steps per visual frame (helps stability at high FPS)
// const SUBSTEPS: usize = 4;

// /// Edge damping width and strength (simple absorbing boundary)
// const DAMP_BORDER: usize = 8;
// const DAMP_STRENGTH: f32 = 0.015; // 0..~0.05 reasonable

// /// Source (soft current Jz) parameters
// const SRC_AMP: f32 = 1.0;
// const SRC_STD_STEPS: f32 = 12.0; // Gaussian width (in time steps)

// #[derive(Resource)]
// struct Grid {
//     nx: usize,
//     ny: usize,
//     dx: f32,
//     dy: f32,
//     dt: f32,
//     // Fields on a Yee grid (TMz): Ez at cell centers, Hx at y-edges, Hy at x-edges
//     ez: Vec<f32>, // size nx*ny
//     hx: Vec<f32>, // size nx*ny
//     hy: Vec<f32>, // size nx*ny
//     // Visualization scaling
//     viz_scale: f32,
//     // Time step counter
//     step: u64,
// }

// impl Grid {
//     fn new(nx: usize, ny: usize, dx: f32, dy: f32) -> Self {
//         let dt = CFL / (C0 * (1.0 / dx.powi(2) + 1.0 / dy.powi(2)).sqrt());
//         Self {
//             nx,
//             ny,
//             dx,
//             dy,
//             dt,
//             ez: vec![0.0; nx * ny],
//             hx: vec![0.0; nx * ny],
//             hy: vec![0.0; nx * ny],
//             viz_scale: 1.5, // tweak for brightness
//             step: 0,
//         }
//     }

//     #[inline]
//     fn idx(&self, i: usize, j: usize) -> usize {
//         j * self.nx + i
//     }
// }

// /// Simple RGBA8 pixel for our texture
// #[repr(C)]
// #[derive(Clone, Copy, Pod, Zeroable)]
// struct Rgba8 {
//     r: u8,
//     g: u8,
//     b: u8,
//     a: u8,
// }

// #[derive(Resource)]
// struct FieldTexture(Handle<Image>);

// fn main() {
//     App::new()
//         .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest())) // crisp pixels
//         .insert_resource(Grid::new(NX, NY, DX, DY))
//         .add_systems(Startup, setup)
//         .add_systems(
//             Update,
//             (fdtd_step_system, upload_texture_system, camera_controls),
//         )
//         .run();
// }

// fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
//     // Create a blank texture for visualizing Ez
//     let mut image = Image::new_fill(
//         Extent3d {
//             width: NX as u32,
//             height: NY as u32,
//             depth_or_array_layers: 1,
//         },
//         TextureDimension::D2,
//         &[0u8; 4],
//         TextureFormat::Rgba8UnormSrgb,
//     );
//     image.texture_descriptor.usage |= bevy::render::render_resource::TextureUsages::COPY_DST;

//     let handle = images.add(image);
//     commands.insert_resource(FieldTexture(handle.clone()));

//     // 2D camera and sprite displaying the texture
//     commands.spawn(Camera2dBundle::default());

//     commands.spawn(SpriteBundle {
//         texture: handle,
//         transform: Transform {
//             translation: Vec3::ZERO,
//             scale: Vec3::new(3.0, 3.0, 1.0), // upscale on screen
//             ..Default::default()
//         },
//         ..Default::default()
//     });

//     info!(
//         "Δt = {:.3e} s, Δx = {:.3e} m, c0Δt/Δx ≈ {:.3}",
//         Grid::new(NX, NY, DX, DY).dt,
//         DX,
//         C0 * Grid::new(NX, NY, DX, DY).dt / DX
//     );
// }

// /// One or more FDTD sub-steps per frame.
// fn fdtd_step_system(mut grid: ResMut<Grid>) {
//     for _ in 0..SUBSTEPS {
//         update_h(&mut grid);
//         inject_source(&mut grid); // soft source enters Ampère's law via Jz term
//         update_e(&mut grid);
//         apply_simple_absorbing(&mut grid);
//         grid.step += 1;
//     }
// }

// /// Update Hx, Hy from Faraday's law: ∇×E = -∂B/∂t  with B = μ H
// fn update_h(grid: &mut Grid) {
//     let nx = grid.nx;
//     let ny = grid.ny;
//     let dt_over_mu_dx = grid.dt / (MU0 * grid.dx);
//     let dt_over_mu_dy = grid.dt / (MU0 * grid.dy);

//     // Yee staggering implicit: we use centered differences
//     // Hx(i,j)^{n+1/2} = Hx(i,j)^{n-1/2} - (dt/μ) * ( ∂Ez/∂y )
//     // Hy(i,j)^{n+1/2} = Hy(i,j)^{n-1/2} + (dt/μ) * ( ∂Ez/∂x )
//     for j in 0..ny - 1 {
//         for i in 0..nx {
//             let idx = grid.idx(i, j);
//             let d_ez_dy = grid.ez[grid.idx(i, j + 1)] - grid.ez[idx];
//             grid.hx[idx] -= dt_over_mu_dy * d_ez_dy;
//         }
//     }
//     for j in 0..ny {
//         for i in 0..nx - 1 {
//             let idx = grid.idx(i, j);
//             let d_ez_dx = grid.ez[grid.idx(i + 1, j)] - grid.ez[idx];
//             grid.hy[idx] += dt_over_mu_dx * d_ez_dx;
//         }
//     }
// }

// /// Update Ez from Ampère–Maxwell: ∇×H = J + ε ∂E/∂t  (we rearrange for Ez^{n+1})
// fn update_e(grid: &mut Grid) {
//     let nx = grid.nx;
//     let ny = grid.ny;
//     let dt_over_eps_dx = grid.dt / (EPS0 * grid.dx);
//     let dt_over_eps_dy = grid.dt / (EPS0 * grid.dy);

//     // Ez(i,j)^{n+1} = Ez(i,j)^n + (dt/ε) * ( ∂Hy/∂x - ∂Hx/∂y ) - (dt/ε) * Jz
//     for j in 1..ny {
//         for i in 1..nx {
//             let idx = grid.idx(i, j);
//             let d_hy_dx = grid.hy[idx] - grid.hy[grid.idx(i - 1, j)];
//             let d_hx_dy = grid.hx[idx] - grid.hx[grid.idx(i, j - 1)];
//             grid.ez[idx] += dt_over_eps_dx * d_hy_dx - dt_over_eps_dy * d_hx_dy;
//         }
//     }

//     // PEC-like boundary (Ez=0) at outermost ring to avoid indexing issues
//     for i in 0..nx {
//         grid.ez[grid.idx(i, 0)] = 0.0;
//         grid.ez[grid.idx(i, ny - 1)] = 0.0;
//     }
//     for j in 0..ny {
//         grid.ez[grid.idx(0, j)] = 0.0;
//         grid.ez[grid.idx(nx - 1, j)] = 0.0;
//     }
// }

// /// Inject a soft Gaussian current source Jz centered in the grid
// fn inject_source(grid: &mut Grid) {
//     let i0 = grid.nx / 2;
//     let j0 = grid.ny / 2;

//     // Time-Gaussian: Jz(t) = A * exp( -((t - t0)/σ)^2 )
//     // Put it into Ampère’s law by adding - (dt/ε) * Jz to Ez update.
//     // Easiest numerically: add equivalent Ez kick directly here (same as -dt*Jz/ε).
//     let t = grid.step as f32;
//     let t0 = 4.0 * SRC_STD_STEPS;
//     let gauss = (-((t - t0).powi(2)) / (2.0 * SRC_STD_STEPS.powi(2))).exp();
//     let ez_kick = -grid.dt * (SRC_AMP * gauss) / EPS0;

//     // Deposit into a small disk for smoothness
//     for dj in -1..=1 {
//         for di in -1..=1 {
//             let i = (i0 as isize + di) as usize;
//             let j = (j0 as isize + dj) as usize;
//             let w = if di == 0 && dj == 0 { 1.0 } else { 0.6 };
//             grid.ez[grid.idx(i, j)] += w * ez_kick;
//         }
//     }
// }

// /// Very simple absorbing boundary: exponentially damp near edges
// fn apply_simple_absorbing(grid: &mut Grid) {
//     let nx = grid.nx;
//     let ny = grid.ny;

//     // Precompute 1D tapers
//     let mut wx = vec![1.0f32; nx];
//     let mut wy = vec![1.0f32; ny];

//     for i in 0..DAMP_BORDER.min(nx / 2) {
//         let f = (i as f32) / (DAMP_BORDER as f32);
//         let w = (1.0 - DAMP_STRENGTH * (1.0 - f)).max(0.0);
//         wx[i] *= w;
//         wx[nx - 1 - i] *= w;
//     }
//     for j in 0..DAMP_BORDER.min(ny / 2) {
//         let f = (j as f32) / (DAMP_BORDER as f32);
//         let w = (1.0 - DAMP_STRENGTH * (1.0 - f)).max(0.0);
//         wy[j] *= w;
//         wy[ny - 1 - j] *= w;
//     }

//     // Apply to Ez only (good enough for demo)
//     for j in 0..ny {
//         for i in 0..nx {
//             let idx = grid.idx(i, j);
//             grid.ez[idx] *= wx[i] * wy[j];
//         }
//     }
// }

// /// Write Ez into the texture as a diverging colormap (simple blue-white-red)
// fn upload_texture_system(
//     grid: Res<Grid>,
//     field_tex: Res<FieldTexture>,
//     mut images: ResMut<Assets<Image>>,
// ) {
//     if let Some(img) = images.get_mut(&field_tex.0) {
//         let mut pixels: Vec<Rgba8> = Vec::with_capacity(grid.nx * grid.ny);

//         // Normalize Ez to [-1, 1] by a fixed visualization scale
//         for j in 0..grid.ny {
//             for i in 0..grid.nx {
//                 let ez = grid.ez[grid.idx(i, j)] * grid.viz_scale;
//                 let x = ez.clamp(-1.0, 1.0);

//                 // Simple diverging map: negative=blue, positive=red, zero=white
//                 let r = ((x.max(0.0)) * 255.0) as u8;
//                 let b = ((-x.max(0.0)) * 255.0) as u8;
//                 let g = (255.0 - ((x.abs()) * 255.0 * 0.5)) // desaturate towards center
//                     .clamp(0.0, 255.0) as u8;

//                 pixels.push(Rgba8 { r, g, b, a: 255 });
//             }
//         }

//         let bytes: &[u8] = bytemuck::cast_slice(&pixels);
//         img.resize(Extent3d {
//             width: grid.nx as u32,
//             height: grid.ny as u32,
//             depth_or_array_layers: 1,
//         });
//         img.data.clear();
//         img.data.extend_from_slice(bytes);
//     }
// }

// /// Basic camera controls: scroll to change scale, drag to move
// fn camera_controls(
//     mut q_cam: Query<(&mut Transform, &mut OrthographicProjection), With<Camera>>,
//     mut scroll_evr: EventReader<MouseWheel>,
//     mouse: Res<ButtonInput<MouseButton>>,
//     windows: Query<&Window>,
//     mut last_pos: Local<Option<Vec2>>,
//     mut cursor_evr: EventReader<CursorMoved>,
// ) {
//     let (mut tf, mut proj) = match q_cam.get_single_mut() {
//         Ok(v) => v,
//         Err(_) => return,
//     };

//     // Zoom
//     for ev in scroll_evr.read() {
//         let factor = if ev.y > 0.0 { 0.9 } else { 1.1 };
//         proj.scale = (proj.scale * factor).clamp(0.2, 10.0);
//     }

//     // Pan
//     let window = match windows.get_single() {
//         Ok(w) => w,
//         Err(_) => return,
//     };
//     for ev in cursor_evr.read() {
//         if mouse.pressed(MouseButton::Left) {
//             if let Some(prev) = *last_pos {
//                 let delta = ev.position - prev;
//                 // Adjust for zoom scale
//                 tf.translation.x += delta.x * proj.scale;
//                 tf.translation.y -= delta.y * proj.scale;
//             }
//         }
//         *last_pos = Some(ev.position);
//     }
// }

use bevy::input::mouse::MouseWheel;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, RenderAssetUsages, TextureDimension, TextureFormat};
use bytemuck::{Pod, Zeroable};

/// Physical constants (SI)
const EPS0: f32 = 8.854_187e-12;
const MU0: f32 = 4.0 * std::f32::consts::PI * 1.0e-7;
fn c0() -> f32 {
    1.0 / (MU0 * EPS0).sqrt()
}

/// Simulation grid
const NX: usize = 300;
const NY: usize = 200;
const DX: f32 = 1.0;
const DY: f32 = 1.0;
const CFL: f32 = 0.5;
const SUBSTEPS: usize = 4;
const DAMP_BORDER: usize = 8;
const DAMP_STRENGTH: f32 = 0.015;
const SRC_AMP: f32 = 1.0;
const SRC_STD_STEPS: f32 = 12.0;

#[derive(Resource)]
struct Grid {
    nx: usize,
    ny: usize,
    dx: f32,
    dy: f32,
    dt: f32,
    ez: Vec<f32>,
    hx: Vec<f32>,
    hy: Vec<f32>,
    viz_scale: f32,
    step: u64,
}

impl Grid {
    fn new(nx: usize, ny: usize, dx: f32, dy: f32) -> Self {
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
    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Rgba8 {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

#[derive(Resource)]
struct FieldTexture(Handle<Image>);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(ImagePlugin::default_nearest()))
        .insert_resource(Grid::new(NX, NY, DX, DY))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (fdtd_step_system, upload_texture_system, camera_controls),
        )
        .run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
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
    image.texture_descriptor.usage |= bevy::render::render_resource::TextureUsages::COPY_DST;

    let handle = images.add(image);
    commands.insert_resource(FieldTexture(handle.clone()));
    commands.spawn(Camera2dBundle::default());
    commands.spawn(SpriteBundle {
        texture: handle,
        transform: Transform::from_scale(Vec3::splat(3.0)),
        ..Default::default()
    });

    let g = Grid::new(NX, NY, DX, DY);
    info!("Δt = {:.3e} s, c0Δt/Δx ≈ {:.3}", g.dt, c0() * g.dt / DX);
}

fn fdtd_step_system(mut grid: ResMut<Grid>) {
    for _ in 0..SUBSTEPS {
        update_h(&mut grid);
        inject_source(&mut grid);
        update_e(&mut grid);
        apply_simple_absorbing(&mut grid);
        grid.step += 1;
    }
}

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

fn inject_source(grid: &mut Grid) {
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

fn upload_texture_system(
    grid: Res<Grid>,
    field_tex: Res<FieldTexture>,
    mut images: ResMut<Assets<Image>>,
) {
    if let Some(img) = images.get_mut(&field_tex.0) {
        let mut pixels: Vec<Rgba8> = Vec::with_capacity(grid.nx * grid.ny);
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let ez = grid.ez[grid.idx(i, j)] * grid.viz_scale;
                let x = ez.clamp(-1.0, 1.0);
                let r = ((x.max(0.0)) * 255.0) as u8;
                let b = ((-x.max(0.0)) * 255.0) as u8;
                let g = (255.0 - ((x.abs()) * 255.0 * 0.5)).clamp(0.0, 255.0) as u8;
                pixels.push(Rgba8 { r, g, b, a: 255 });
            }
        }
        img.resize(Extent3d {
            width: grid.nx as u32,
            height: grid.ny as u32,
            depth_or_array_layers: 1,
        });
        img.data.clear();
        img.data.extend_from_slice(bytemuck::cast_slice(&pixels));
    }
}

fn camera_controls(
    mut q_cam: Query<(&mut Transform, &mut OrthographicProjection), With<Camera>>,
    mut scroll_evr: EventReader<MouseWheel>,
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    mut last_pos: Local<Option<Vec2>>,
    mut cursor_evr: EventReader<CursorMoved>,
) {
    if let Ok((mut tf, mut proj)) = q_cam.get_single_mut() {
        for ev in scroll_evr.read() {
            let factor = if ev.y > 0.0 { 0.9 } else { 1.1 };
            proj.scale = (proj.scale * factor).clamp(0.2, 10.0);
        }
        if let Ok(window) = windows.get_single() {
            for ev in cursor_evr.read() {
                if mouse.pressed(MouseButton::Left) {
                    if let Some(prev) = *last_pos {
                        let delta = ev.position - prev;
                        tf.translation.x += delta.x * proj.scale;
                        tf.translation.y -= delta.y * proj.scale;
                    }
                }
                *last_pos = Some(ev.position);
            }
        }
    }
}

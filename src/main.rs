mod game;

use anyhow::Result;
use game::{run_game, CaretDirection, Changes, GameState};
use notify_debouncer_mini::{new_debouncer, notify::*, DebounceEventResult};
use rand::{seq::SliceRandom, Rng};
use roxmltree::{Document, Node};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    fmt::Debug,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
    time::{Duration, Instant},
};
use vello::{
    kurbo::{
        Affine, BezPath, CubicBez, Line, ParamCurve, PathEl, PathSeg, Point, QuadBez, Rect, Size,
        Stroke, Vec2,
    },
    peniko::{Blob, Brush, BrushRef, Color, Fill, Font, Format, Image, StyleRef},
    skrifa::{raw::FontRef, MetadataProvider},
    util::{RenderContext, RenderSurface},
    wgpu, AaConfig, Glyph, Renderer, RendererOptions, Scene,
};
use vello_encoding::BumpAllocators;
use wgpu_profiler::GpuTimerQueryResult;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    keyboard::*,
    window::{Window, WindowAttributes},
};

fn default_threads() -> usize {
    #[cfg(target_os = "macos")]
    return 1;
    #[cfg(not(target_os = "macos"))]
    return 0;
}

struct RenderState<'s> {
    // SAFETY: We MUST drop the surface before the `window`, so the fields
    // must be in this order
    surface: RenderSurface<'s>,
    window: Arc<Window>,
}

// TODO: Make this set configurable through the command line
// Alternatively, load anti-aliasing shaders on demand/asynchronously
const AA_CONFIGS: [AaConfig; 3] = [AaConfig::Area, AaConfig::Msaa8, AaConfig::Msaa16];

struct VelloApp<'s> {
    context: RenderContext,
    renderers: Vec<Option<Renderer>>,
    state: Option<RenderState<'s>>,
    // Whilst suspended, we drop `render_state`, but need to keep the same window.
    // If render_state exists, we must store the window in it, to maintain drop order
    cached_window: Option<Arc<Window>>,

    use_cpu: bool,
    num_init_threads: usize,

    scene: Scene,
    fragment: Scene,
    simple_text: SimpleText,
    images: ImageCache,
    stats: Stats,
    stats_shown: bool,

    base_color: Option<Color>,
    async_pipeline: bool,

    // Currently not updated in wasm builds
    #[allow(unused_mut)]
    scene_complexity: Option<BumpAllocators>,

    complexity_shown: bool,
    vsync_on: bool,

    gpu_profiling_on: bool,
    profile_stored: Option<Vec<wgpu_profiler::GpuTimerQueryResult>>,
    profile_taken: Instant,

    // We allow cycling through AA configs in either direction, so use a signed index
    aa_config_ix: i32,

    frame_start_time: Instant,
    start: Instant,

    prev_instant: Instant,

    touch_state: TouchState,
    // navigation_fingers are fingers which are used in the navigation 'zone' at the bottom
    // of the screen. This ensures that one press on the screen doesn't have multiple actions
    navigation_fingers: HashSet<u64>,
    mouse_down: bool,
    prior_position: Option<Vec2>,

    modifiers: ModifiersState,

    debug: vello::low_level::DebugLayers,

    current_state: GameState,
    caret_direction: CaretDirection,
    restart: bool,
}

impl<'s> ApplicationHandler<UserEvent> for VelloApp<'s> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let Option::None = self.state else {
            return;
        };
        let window = self
            .cached_window
            .take()
            .unwrap_or_else(|| Arc::new(event_loop.create_window(window_attributes()).unwrap()));
        let size = window.inner_size();
        let present_mode = if self.vsync_on {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        let surface_future =
            self.context
                .create_surface(window.clone(), size.width, size.height, present_mode);
        // We need to block here, in case a Suspended event appeared
        let surface = pollster::block_on(surface_future).expect("Error creating surface");
        self.state = {
            let render_state = RenderState { window, surface };
            self.renderers
                .resize_with(self.context.devices.len(), || None);
            let id = render_state.surface.dev_id;
            self.renderers[id].get_or_insert_with(|| {
                let start = Instant::now();
                let mut renderer = Renderer::new(
                    &self.context.devices[id].device,
                    RendererOptions {
                        surface_format: Some(render_state.surface.format),
                        use_cpu: self.use_cpu,
                        antialiasing_support: AA_CONFIGS.iter().copied().collect(),
                        num_init_threads: NonZeroUsize::new(self.num_init_threads),
                    },
                )
                .map_err(|e| {
                    // Pretty-print any renderer creation error using Display formatting before unwrapping.
                    anyhow::format_err!("{e}")
                })
                .expect("Failed to create renderer");
                log::info!("Creating renderer {id} took {:?}", start.elapsed());
                renderer
                    .profiler
                    .change_settings(wgpu_profiler::GpuProfilerSettings {
                        enable_timer_queries: self.gpu_profiling_on,
                        enable_debug_groups: self.gpu_profiling_on,
                        ..Default::default()
                    })
                    .expect("Not setting max_num_pending_frames");
                renderer
            });
            Some(render_state)
        };
        event_loop.set_control_flow(ControlFlow::Poll);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(render_state) = &mut self.state else {
            return;
        };
        if render_state.window.id() != window_id {
            return;
        }
        let _span = if !matches!(event, WindowEvent::RedrawRequested) {
            Some(tracing::trace_span!("Handling window event", ?event).entered())
        } else {
            None
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::ModifiersChanged(m) => self.modifiers = m.state(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.logical_key.as_ref() {
                        Key::Character(char) => {
                            // TODO: Have a more principled way of handling modifiers on keypress
                            // see e.g. https://xi.zulipchat.com/#narrow/stream/351333-glazier/topic/Keyboard.20shortcuts
                            let char = char.to_lowercase();
                            match char.as_str() {
                                "r" => {
                                    self.restart = true;
                                }
                                "q" => {
                                    self.caret_direction = CaretDirection::Left;
                                }
                                "e" => {
                                    self.caret_direction = CaretDirection::Right;
                                }
                                "s" => {
                                    self.stats_shown = !self.stats_shown;
                                }
                                "d" => {
                                    self.complexity_shown = !self.complexity_shown;
                                }
                                "c" => {
                                    self.stats.clear_min_and_max();
                                }
                                "m" => {
                                    self.aa_config_ix = if self.modifiers.shift_key() {
                                        self.aa_config_ix.saturating_sub(1)
                                    } else {
                                        self.aa_config_ix.saturating_add(1)
                                    };
                                }
                                "p" => {
                                    if let Some(renderer) =
                                        &self.renderers[render_state.surface.dev_id]
                                    {
                                        store_profiling(renderer, &self.profile_stored);
                                    }
                                }
                                "g" => {
                                    self.gpu_profiling_on = !self.gpu_profiling_on;
                                    if let Some(renderer) =
                                        &mut self.renderers[render_state.surface.dev_id]
                                    {
                                        renderer
                                            .profiler
                                            .change_settings(wgpu_profiler::GpuProfilerSettings {
                                                enable_timer_queries: self.gpu_profiling_on,
                                                enable_debug_groups: self.gpu_profiling_on,
                                                ..Default::default()
                                            })
                                            .expect("Not setting max_num_pending_frames");
                                    }
                                }
                                "v" => {
                                    self.vsync_on = !self.vsync_on;
                                    self.context.set_present_mode(
                                        &mut render_state.surface,
                                        if self.vsync_on {
                                            wgpu::PresentMode::AutoVsync
                                        } else {
                                            wgpu::PresentMode::AutoNoVsync
                                        },
                                    );
                                }
                                debug_layer @ ("1" | "2" | "3" | "4") => {
                                    match debug_layer {
                                        "1" => {
                                            self.debug.toggle(
                                                vello::low_level::DebugLayers::BOUNDING_BOXES,
                                            );
                                        }
                                        "2" => {
                                            self.debug.toggle(
                                                vello::low_level::DebugLayers::LINESOUP_SEGMENTS,
                                            );
                                        }
                                        "3" => {
                                            self.debug.toggle(
                                                vello::low_level::DebugLayers::LINESOUP_POINTS,
                                            );
                                        }
                                        "4" => {
                                            self.debug
                                                .toggle(vello::low_level::DebugLayers::VALIDATION);
                                        }
                                        _ => unreachable!(),
                                    }
                                    if !self.debug.is_empty() && !self.async_pipeline {
                                        log::warn!("Debug Layers won't work without using `--async-pipeline`. Requested {:?}", self.debug);
                                    }
                                }
                                _ => {}
                            }
                        }
                        Key::Named(NamedKey::Escape) => event_loop.exit(),
                        _ => {}
                    }
                } else {
                    self.caret_direction = CaretDirection::None;
                }
            }
            WindowEvent::Touch(touch) => {
                match touch.phase {
                    TouchPhase::Started => {
                        // We reserve the bottom third of the screen for navigation
                        // This also prevents strange effects whilst using the navigation gestures on Android
                        // TODO: How do we know what the client area is? Winit seems to just give us the
                        // full screen
                        // TODO: Render a display of the navigation regions. We don't do
                        // this currently because we haven't researched how to determine when we're
                        // in a touch context (i.e. Windows/Linux/MacOS with a touch screen could
                        // also be using mouse/keyboard controls)
                        // Note that winit's rendering is y-down
                        if let Some(RenderState { surface, .. }) = &self.state {
                            if touch.location.y > surface.config.height as f64 * 2. / 3. {
                                self.navigation_fingers.insert(touch.id);
                            }
                        }
                    }
                    TouchPhase::Ended | TouchPhase::Cancelled => {
                        // We intentionally ignore the result here
                        self.navigation_fingers.remove(&touch.id);
                    }
                    TouchPhase::Moved => (),
                }
                // See documentation on navigation_fingers
                if !self.navigation_fingers.contains(&touch.id) {
                    self.touch_state.add_event(&touch);
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(RenderState { surface, window }) = &mut self.state {
                    self.context
                        .resize_surface(surface, size.width, size.height);
                    window.request_redraw();
                };
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_down = state == ElementState::Pressed;
                }
            }
            WindowEvent::CursorLeft { .. } => {
                self.prior_position = None;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let position = Vec2::new(position.x, position.y);
                self.prior_position = Some(position);
            }
            WindowEvent::RedrawRequested => {
                let _rendering_span = tracing::trace_span!("Actioning Requested Redraw").entered();
                let encoding_span = tracing::trace_span!("Encoding scene").entered();

                let Some(RenderState { surface, .. }) = &self.state else {
                    return;
                };
                let width = surface.config.width;
                let height = surface.config.height;
                let device_handle = &self.context.devices[surface.dev_id];
                let snapshot = self.stats.snapshot();

                // Allow looping forever
                self.aa_config_ix = self.aa_config_ix.rem_euclid(AA_CONFIGS.len() as i32);

                self.fragment.reset();
                let scene_params = SceneParams {
                    time: self.start.elapsed().as_secs_f64(),
                    text: &mut self.simple_text,
                    images: &mut self.images,
                    resolution: None,
                    base_color: None,
                    interactive: true,
                };

                let current_state = std::mem::take(&mut self.current_state);
                let prev_instant = self.prev_instant.clone();
                let now = Instant::now();

                let (state, scene) = run_game(
                    current_state,
                    Changes {
                        prev_instant,
                        now,
                        caret_direction: self.caret_direction,
                        restart: self.restart,
                    },
                );

                self.restart = false;
                self.current_state = state;
                self.scene = scene;
                self.prev_instant = now;

                // If the user specifies a base color in the CLI we use that. Otherwise we use any
                // color specified by the scene. The default is black.
                let base_color = self
                    .base_color
                    .or(scene_params.base_color)
                    .unwrap_or(Color::BLACK);
                let antialiasing_method = AA_CONFIGS[self.aa_config_ix as usize];
                let render_params = vello::RenderParams {
                    base_color,
                    width,
                    height,
                    antialiasing_method,
                };
                let mut transform = Affine::IDENTITY;
                if let Some(resolution) = scene_params.resolution {
                    // Automatically scale the rendering to fill as much of the window as possible
                    // TODO: Apply svg view_box, somehow
                    let factor = Vec2::new(width as f64, height as f64);
                    let scale_factor = (factor.x / resolution.x).min(factor.y / resolution.y);
                    transform *= Affine::scale(scale_factor);
                }
                self.scene.append(&self.fragment, Some(transform));
                if self.stats_shown {
                    snapshot.draw_layer(
                        &mut self.scene,
                        scene_params.text,
                        width as f64,
                        height as f64,
                        self.stats.samples(),
                        self.complexity_shown
                            .then_some(self.scene_complexity)
                            .flatten(),
                        self.vsync_on,
                        antialiasing_method,
                    );
                    if let Some(profiling_result) = self.renderers[surface.dev_id]
                        .as_mut()
                        .and_then(|it| it.profile_result.take())
                    {
                        if self.profile_stored.is_none()
                            || self.profile_taken.elapsed() > Duration::from_secs(1)
                        {
                            self.profile_stored = Some(profiling_result);
                            self.profile_taken = Instant::now();
                        }
                    }
                    if let Some(profiling_result) = self.profile_stored.as_ref() {
                        draw_gpu_profiling(
                            &mut self.scene,
                            scene_params.text,
                            width as f64,
                            height as f64,
                            profiling_result,
                        );
                    }
                }
                drop(encoding_span);
                let texture_span = tracing::trace_span!("Getting texture").entered();
                let surface_texture = surface
                    .surface
                    .get_current_texture()
                    .expect("failed to get surface texture");

                drop(texture_span);
                let render_span = tracing::trace_span!("Dispatching render").entered();
                self.renderers[surface.dev_id]
                    .as_mut()
                    .unwrap()
                    .render_to_surface(
                        &device_handle.device,
                        &device_handle.queue,
                        &self.scene,
                        &surface_texture,
                        &render_params,
                    )
                    .expect("failed to render to surface");
                surface_texture.present();
                drop(render_span);

                {
                    let _poll_aspan = tracing::trace_span!("Polling wgpu device").entered();
                    device_handle.device.poll(wgpu::Maintain::Poll);
                }
                let new_time = Instant::now();
                self.stats.add_sample(Sample {
                    frame_time_us: (new_time - self.frame_start_time).as_micros() as u64,
                });
                self.frame_start_time = new_time;
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.touch_state.end_frame();

        if let Some(render_state) = &mut self.state {
            render_state.window.request_redraw();
        }
    }

    fn user_event(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop, event: UserEvent) {
        match event {
            #[cfg(not(any(target_arch = "wasm32", target_os = "android")))]
            UserEvent::HotReload => {
                let Some(render_state) = &mut self.state else {
                    return;
                };
                let device_handle = &self.context.devices[render_state.surface.dev_id];
                log::info!("==============\nReloading shaders");
                let start = Instant::now();
                let result = self.renderers[render_state.surface.dev_id]
                    .as_mut()
                    .unwrap()
                    .reload_shaders(&device_handle.device);
                // We know that the only async here (`pop_error_scope`) is actually sync, so blocking is fine
                match pollster::block_on(result) {
                    Ok(_) => log::info!("Reloading took {:?}", start.elapsed()),
                    Err(e) => log::error!("Failed to reload shaders: {e}"),
                }
            }
        }
    }

    fn suspended(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        log::info!("Suspending");
        #[cfg(not(target_arch = "wasm32"))]
        // When we suspend, we need to remove the `wgpu` Surface
        if let Some(render_state) = self.state.take() {
            self.cached_window = Some(render_state.window);
        }
        event_loop.set_control_flow(ControlFlow::Wait);
    }
}

fn run(
    event_loop: EventLoop<UserEvent>,
    render_cx: RenderContext,
    #[cfg(target_arch = "wasm32")] render_state: RenderState,
) {
    use winit::keyboard::*;

    let renderers: Vec<Option<Renderer>> = vec![];

    let render_state = None::<RenderState>;

    let debug = vello::low_level::DebugLayers::none();

    let mut app = VelloApp {
        context: render_cx,
        renderers,
        state: render_state,
        cached_window: None,
        use_cpu: false,
        num_init_threads: default_threads(),
        scene: Scene::new(),
        fragment: Scene::new(),
        simple_text: SimpleText::new(),
        images: ImageCache::new(),
        stats: Stats::new(),
        stats_shown: false,
        base_color: Some(Color::WHITE),
        async_pipeline: false,
        scene_complexity: None,
        complexity_shown: false,
        vsync_on: true,

        gpu_profiling_on: true,
        profile_stored: None,
        profile_taken: Instant::now(),
        prev_instant: Instant::now(),
        caret_direction: CaretDirection::None,

        aa_config_ix: 0,

        frame_start_time: Instant::now(),
        start: Instant::now(),

        touch_state: TouchState::new(),
        navigation_fingers: HashSet::new(),
        mouse_down: false,
        prior_position: None,
        modifiers: ModifiersState::default(),
        debug,
        current_state: Default::default(),
        restart: false,
    };

    event_loop.run_app(&mut app).expect("run to completion");
}

/// A function extracted to fix rustfmt
fn store_profiling(
    renderer: &Renderer,
    profile_stored: &Option<Vec<wgpu_profiler::GpuTimerQueryResult>>,
) {
    if let Some(profile_result) = &renderer.profile_result.as_ref().or(profile_stored.as_ref()) {
        // There can be empty results if the required features aren't supported
        if !profile_result.is_empty() {
            let path = std::path::Path::new("trace.json");
            match wgpu_profiler::chrometrace::write_chrometrace(path, profile_result) {
                Ok(()) => {
                    println!("Wrote trace to path {path:?}");
                }
                Err(e) => {
                    log::warn!("Failed to write trace {e}");
                }
            }
        }
    }
}

fn window_attributes() -> WindowAttributes {
    Window::default_attributes()
        .with_inner_size(LogicalSize::new(1800, 1000))
        .with_resizable(true)
        .with_title("Vello demo")
}

#[derive(Debug)]
enum UserEvent {
    HotReload,
}

fn main() -> anyhow::Result<()> {
    // TODO: initializing both env_logger and console_logger fails on wasm.
    // Figure out a more principled approach.
    env_logger::builder()
        .format_timestamp(Some(env_logger::TimestampPrecision::Millis))
        .filter_level(log::LevelFilter::Warn)
        .init();
    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    #[allow(unused_mut)]
    let mut render_cx = RenderContext::new();
    let proxy = event_loop.create_proxy();
    let _keep = hot_reload(move || proxy.send_event(UserEvent::HotReload).ok().map(drop));

    run(event_loop, render_cx);
    Ok(())
}

pub fn hot_reload(mut f: impl FnMut() -> Option<()> + Send + 'static) -> Result<impl Sized> {
    let mut debouncer = new_debouncer(
        Duration::from_millis(500),
        move |res: DebounceEventResult| match res {
            Ok(_) => f().unwrap(),
            Err(e) => println!("Hot reloading file watching failed: {e:?}"),
        },
    )?;

    debouncer.watcher().watch(
        vello_shaders::compile::shader_dir().as_path(),
        // We currently don't support hot reloading the imports, so don't recurse into there
        RecursiveMode::NonRecursive,
    )?;
    Ok(debouncer)
}

/// All you probably need to know about a multi-touch gesture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultiTouchInfo {
    /// Number of touches (fingers) on the surface. Value is ≥ 2 since for a single touch no
    /// [`MultiTouchInfo`] is created.
    pub num_touches: usize,

    /// Proportional zoom factor (pinch gesture).
    /// * `zoom = 1`: no change
    /// * `zoom < 1`: pinch together
    /// * `zoom > 1`: pinch spread
    pub zoom_delta: f64,

    /// 2D non-proportional zoom factor (pinch gesture).
    ///
    /// For horizontal pinches, this will return `[z, 1]`,
    /// for vertical pinches this will return `[1, z]`,
    /// and otherwise this will return `[z, z]`,
    /// where `z` is the zoom factor:
    /// * `zoom = 1`: no change
    /// * `zoom < 1`: pinch together
    /// * `zoom > 1`: pinch spread
    pub zoom_delta_2d: Vec2,

    /// Rotation in radians. Moving fingers around each other will change this value. This is a
    /// relative value, comparing the orientation of fingers in the current frame with the previous
    /// frame. If all fingers are resting, this value is `0.0`.
    pub rotation_delta: f64,

    /// Relative movement (comparing previous frame and current frame) of the average position of
    /// all touch points. Without movement this value is `Vec2::ZERO`.
    ///
    /// Note that this may not necessarily be measured in screen points (although it _will_ be for
    /// most mobile devices). In general (depending on the touch device), touch coordinates cannot
    /// be directly mapped to the screen. A touch always is considered to start at the position of
    /// the pointer, but touch movement is always measured in the units delivered by the device,
    /// and may depend on hardware and system settings.
    pub translation_delta: Vec2,
    pub zoom_centre: Point,
}

/// The current state (for a specific touch device) of touch events and gestures.
#[derive(Clone)]
pub struct TouchState {
    /// Active touches, if any.
    ///
    /// Touch id is the unique identifier of the touch. It is valid as long as the finger/pen
    /// touches the surface. The next touch will receive a new unique id.
    ///
    /// Refer to [`ActiveTouch`].
    active_touches: BTreeMap<u64, ActiveTouch>,

    /// If a gesture has been recognized (i.e. when exactly two fingers touch the surface), this
    /// holds state information
    gesture_state: Option<GestureState>,

    added_or_removed_touches: bool,
}

#[derive(Clone, Debug)]
struct GestureState {
    pinch_type: PinchType,
    previous: Option<DynGestureState>,
    current: DynGestureState,
}

/// Gesture data that can change over time
#[derive(Clone, Copy, Debug)]
struct DynGestureState {
    /// used for proportional zooming
    avg_distance: f64,
    /// used for non-proportional zooming
    avg_abs_distance2: Vec2,
    avg_pos: Point,
    heading: f64,
}

/// Describes an individual touch (finger or digitizer) on the touch surface. Instances exist as
/// long as the finger/pen touches the surface.
#[derive(Clone, Copy, Debug)]
struct ActiveTouch {
    /// Current position of this touch, in device coordinates (not necessarily screen position)
    pos: Point,
}

impl TouchState {
    pub fn new() -> Self {
        Self {
            active_touches: Default::default(),
            gesture_state: None,
            added_or_removed_touches: false,
        }
    }

    pub fn add_event(&mut self, event: &Touch) {
        let pos = Point::new(event.location.x, event.location.y);
        match event.phase {
            TouchPhase::Started => {
                self.active_touches.insert(event.id, ActiveTouch { pos });
                self.added_or_removed_touches = true;
            }
            TouchPhase::Moved => {
                if let Some(touch) = self.active_touches.get_mut(&event.id) {
                    touch.pos = Point::new(event.location.x, event.location.y);
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.active_touches.remove(&event.id);
                self.added_or_removed_touches = true;
            }
        }
    }

    pub fn end_frame(&mut self) {
        // This needs to be called each frame, even if there are no new touch events.
        // Otherwise, we would send the same old delta information multiple times:
        self.update_gesture();

        if self.added_or_removed_touches {
            // Adding or removing fingers makes the average values "jump". We better forget
            // about the previous values, and don't create delta information for this frame:
            if let Some(ref mut state) = &mut self.gesture_state {
                state.previous = None;
            }
        }
        self.added_or_removed_touches = false;
    }

    pub fn info(&self) -> Option<MultiTouchInfo> {
        self.gesture_state.as_ref().map(|state| {
            // state.previous can be `None` when the number of simultaneous touches has just
            // changed. In this case, we take `current` as `previous`, pretending that there
            // was no change for the current frame.
            let state_previous = state.previous.unwrap_or(state.current);

            let zoom_delta = if self.active_touches.len() > 1 {
                state.current.avg_distance / state_previous.avg_distance
            } else {
                1.
            };

            let zoom_delta2 = if self.active_touches.len() > 1 {
                match state.pinch_type {
                    PinchType::Horizontal => Vec2::new(
                        state.current.avg_abs_distance2.x / state_previous.avg_abs_distance2.x,
                        1.0,
                    ),
                    PinchType::Vertical => Vec2::new(
                        1.0,
                        state.current.avg_abs_distance2.y / state_previous.avg_abs_distance2.y,
                    ),
                    PinchType::Proportional => Vec2::new(zoom_delta, zoom_delta),
                }
            } else {
                Vec2::new(1.0, 1.0)
            };

            MultiTouchInfo {
                num_touches: self.active_touches.len(),
                zoom_delta,
                zoom_delta_2d: zoom_delta2,
                zoom_centre: state.current.avg_pos,
                rotation_delta: (state.current.heading - state_previous.heading),
                translation_delta: state.current.avg_pos - state_previous.avg_pos,
            }
        })
    }

    fn update_gesture(&mut self) {
        if let Some(dyn_state) = self.calc_dynamic_state() {
            if let Some(ref mut state) = &mut self.gesture_state {
                // updating an ongoing gesture
                state.previous = Some(state.current);
                state.current = dyn_state;
            } else {
                // starting a new gesture
                self.gesture_state = Some(GestureState {
                    pinch_type: PinchType::classify(&self.active_touches),
                    previous: None,
                    current: dyn_state,
                });
            }
        } else {
            // the end of a gesture (if there is any)
            self.gesture_state = None;
        }
    }

    /// `None` if less than two fingers
    fn calc_dynamic_state(&self) -> Option<DynGestureState> {
        let num_touches = self.active_touches.len();
        if num_touches == 0 {
            return None;
        }
        let mut state = DynGestureState {
            avg_distance: 0.0,
            avg_abs_distance2: Vec2::ZERO,
            avg_pos: Point::ZERO,
            heading: 0.0,
        };
        let num_touches_recip = 1. / num_touches as f64;

        // first pass: calculate force and center of touch positions:
        for touch in self.active_touches.values() {
            state.avg_pos.x += touch.pos.x;
            state.avg_pos.y += touch.pos.y;
        }
        state.avg_pos.x *= num_touches_recip;
        state.avg_pos.y *= num_touches_recip;

        // second pass: calculate distances from center:
        for touch in self.active_touches.values() {
            state.avg_distance += state.avg_pos.distance(touch.pos);
            state.avg_abs_distance2.x += (state.avg_pos.x - touch.pos.x).abs();
            state.avg_abs_distance2.y += (state.avg_pos.y - touch.pos.y).abs();
        }
        state.avg_distance *= num_touches_recip;
        state.avg_abs_distance2 *= num_touches_recip;

        // Calculate the direction from the first touch to the center position.
        // This is not the perfect way of calculating the direction if more than two fingers
        // are involved, but as long as all fingers rotate more or less at the same angular
        // velocity, the shortcomings of this method will not be noticed. One can see the
        // issues though, when touching with three or more fingers, and moving only one of them
        // (it takes two hands to do this in a controlled manner). A better technique would be
        // to store the current and previous directions (with reference to the center) for each
        // touch individually, and then calculate the average of all individual changes in
        // direction. But this approach cannot be implemented locally in this method, making
        // everything a bit more complicated.
        let first_touch = self.active_touches.values().next().unwrap();
        state.heading = (state.avg_pos - first_touch.pos).atan2();

        Some(state)
    }
}

impl Debug for TouchState {
    // This outputs less clutter than `#[derive(Debug)]`:
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (id, touch) in &self.active_touches {
            f.write_fmt(format_args!("#{:?}: {:#?}\n", id, touch))?;
        }
        f.write_fmt(format_args!("gesture: {:#?}\n", self.gesture_state))?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
enum PinchType {
    Horizontal,
    Vertical,
    Proportional,
}

impl PinchType {
    fn classify(touches: &BTreeMap<u64, ActiveTouch>) -> Self {
        // For non-proportional 2d zooming:
        // If the user is pinching with two fingers that have roughly the same Y coord,
        // then the Y zoom is unstable and should be 1.
        // Similarly, if the fingers are directly above/below each other,
        // we should only zoom on the Y axis.
        // If the fingers are roughly on a diagonal, we revert to the proportional zooming.

        if touches.len() == 2 {
            let mut touches = touches.values();
            let t0 = touches.next().unwrap().pos;
            let t1 = touches.next().unwrap().pos;

            let dx = (t0.x - t1.x).abs();
            let dy = (t0.y - t1.y).abs();

            if dx > 3.0 * dy {
                Self::Horizontal
            } else if dy > 3.0 * dx {
                Self::Vertical
            } else {
                Self::Proportional
            }
        } else {
            Self::Proportional
        }
    }
}

const SLIDING_WINDOW_SIZE: usize = 100;

#[derive(Debug)]
pub struct Snapshot {
    pub fps: f64,
    pub frame_time_ms: f64,
    pub frame_time_min_ms: f64,
    pub frame_time_max_ms: f64,
}

impl Snapshot {
    #[allow(clippy::too_many_arguments)]
    pub fn draw_layer<'a, T>(
        &self,
        scene: &mut Scene,
        text: &mut SimpleText,
        viewport_width: f64,
        viewport_height: f64,
        samples: T,
        bump: Option<BumpAllocators>,
        vsync: bool,
        aa_config: AaConfig,
    ) where
        T: Iterator<Item = &'a u64>,
    {
        let width = (viewport_width * 0.4).clamp(200., 600.);
        let height = width * 0.7;
        let x_offset = viewport_width - width;
        let y_offset = viewport_height - height;
        let offset = Affine::translate((x_offset, y_offset));

        // Draw the background
        scene.fill(
            Fill::NonZero,
            offset,
            &Brush::Solid(Color::rgba8(0, 0, 0, 200)),
            None,
            &Rect::new(0., 0., width, height),
        );

        let mut labels = vec![
            format!("Frame Time: {:.2} ms", self.frame_time_ms),
            format!("Frame Time (min): {:.2} ms", self.frame_time_min_ms),
            format!("Frame Time (max): {:.2} ms", self.frame_time_max_ms),
            format!("VSync: {}", if vsync { "on" } else { "off" }),
            format!(
                "AA method: {}",
                match aa_config {
                    AaConfig::Area => "Analytic Area",
                    AaConfig::Msaa16 => "16xMSAA",
                    AaConfig::Msaa8 => "8xMSAA",
                }
            ),
            format!("Resolution: {viewport_width}x{viewport_height}"),
        ];
        if let Some(bump) = &bump {
            if bump.failed >= 1 {
                labels.push("Allocation Failed!".into());
            }
            labels.push(format!("binning: {}", bump.binning));
            labels.push(format!("ptcl: {}", bump.ptcl));
            labels.push(format!("tile: {}", bump.tile));
            labels.push(format!("segments: {}", bump.segments));
            labels.push(format!("blend: {}", bump.blend));
        }

        // height / 2 is dedicated to the text labels and the rest is filled by the bar graph.
        let text_height = height * 0.5 / (1 + labels.len()) as f64;
        let left_margin = width * 0.01;
        let text_size = (text_height * 0.9) as f32;
        for (i, label) in labels.iter().enumerate() {
            text.add(
                scene,
                None,
                text_size,
                Some(&Brush::Solid(Color::WHITE)),
                offset * Affine::translate((left_margin, (i + 1) as f64 * text_height)),
                label,
            );
        }
        text.add(
            scene,
            None,
            text_size,
            Some(&Brush::Solid(Color::WHITE)),
            offset * Affine::translate((width * 0.67, text_height)),
            &format!("FPS: {:.2}", self.fps),
        );

        // Plot the samples with a bar graph
        use PathEl::*;
        let left_padding = width * 0.05; // Left padding for the frame time marker text.
        let graph_max_height = height * 0.5;
        let graph_max_width = width - 2. * left_margin - left_padding;
        let left_margin_padding = left_margin + left_padding;
        let bar_extent = graph_max_width / (SLIDING_WINDOW_SIZE as f64);
        let bar_width = bar_extent * 0.4;
        let bar = [
            MoveTo((0., graph_max_height).into()),
            LineTo((0., 0.).into()),
            LineTo((bar_width, 0.).into()),
            LineTo((bar_width, graph_max_height).into()),
        ];
        // We determine the scale of the graph based on the maximum sampled frame time unless it's
        // greater than 3x the current average. In that case we cap the max scale at 4/3 * the
        // current average (rounded up to the nearest multiple of 5ms). This allows the scale to
        // adapt to the most recent sample set as relying on the maximum alone can make the
        // displayed samples to look too small in the presence of spikes/fluctuation without
        // manually resetting the max sample.
        let display_max = if self.frame_time_max_ms > 3. * self.frame_time_ms {
            round_up((1.33334 * self.frame_time_ms) as usize, 5) as f64
        } else {
            self.frame_time_max_ms
        };
        for (i, sample) in samples.enumerate() {
            let t = offset * Affine::translate((i as f64 * bar_extent, graph_max_height));
            // The height of each sample is based on its ratio to the maximum observed frame time.
            let sample_ms = ((*sample as f64) * 0.001).min(display_max);
            let h = sample_ms / display_max;
            let s = Affine::scale_non_uniform(1., -h);
            #[allow(clippy::match_overlapping_arm)]
            let color = match *sample {
                ..=16_667 => Color::rgb8(100, 143, 255),
                ..=33_334 => Color::rgb8(255, 176, 0),
                _ => Color::rgb8(220, 38, 127),
            };
            scene.fill(
                Fill::NonZero,
                t * Affine::translate((
                    left_margin_padding,
                    (1 + labels.len()) as f64 * text_height,
                )) * s,
                color,
                None,
                &bar,
            );
        }
        // Draw horizontal lines to mark 8.33ms, 16.33ms, and 33.33ms
        let marker = [
            MoveTo((0., graph_max_height).into()),
            LineTo((graph_max_width, graph_max_height).into()),
        ];
        let thresholds = [8.33, 16.66, 33.33];
        let thres_text_height = graph_max_height * 0.05;
        let thres_text_height_2 = thres_text_height * 0.5;
        for t in thresholds.iter().filter(|&&t| t < display_max) {
            let y = t / display_max;
            text.add(
                scene,
                None,
                thres_text_height as f32,
                Some(&Brush::Solid(Color::WHITE)),
                offset
                    * Affine::translate((
                        left_margin,
                        (2. - y) * graph_max_height + thres_text_height_2,
                    )),
                &format!("{}", t),
            );
            scene.stroke(
                &Stroke::new(graph_max_height * 0.01),
                offset * Affine::translate((left_margin_padding, (1. - y) * graph_max_height)),
                Color::WHITE,
                None,
                &marker,
            );
        }
    }
}

pub struct Sample {
    pub frame_time_us: u64,
}

pub struct Stats {
    count: usize,
    sum: u64,
    min: u64,
    max: u64,
    samples: VecDeque<u64>,
}

impl Stats {
    pub fn new() -> Stats {
        Stats {
            count: 0,
            sum: 0,
            min: u64::MAX,
            max: u64::MIN,
            samples: VecDeque::with_capacity(SLIDING_WINDOW_SIZE),
        }
    }

    pub fn samples(&self) -> impl Iterator<Item = &u64> {
        self.samples.iter()
    }

    pub fn snapshot(&self) -> Snapshot {
        let frame_time_ms = (self.sum as f64 / self.count as f64) * 0.001;
        let fps = 1000. / frame_time_ms;
        Snapshot {
            fps,
            frame_time_ms,
            frame_time_min_ms: self.min as f64 * 0.001,
            frame_time_max_ms: self.max as f64 * 0.001,
        }
    }

    pub fn clear_min_and_max(&mut self) {
        self.min = u64::MAX;
        self.max = u64::MIN;
    }

    pub fn add_sample(&mut self, sample: Sample) {
        let oldest = if self.count < SLIDING_WINDOW_SIZE {
            self.count += 1;
            None
        } else {
            self.samples.pop_front()
        };
        let micros = sample.frame_time_us;
        self.sum += micros;
        self.samples.push_back(micros);
        if let Some(oldest) = oldest {
            self.sum -= oldest;
        }
        self.min = self.min.min(micros);
        self.max = self.max.max(micros);
    }
}

fn round_up(n: usize, f: usize) -> usize {
    n - 1 - (n - 1) % f + f
}

fn profiles_are_empty(profiles: &[GpuTimerQueryResult]) -> bool {
    profiles.iter().all(|p| p.time.is_none())
}

pub fn draw_gpu_profiling(
    scene: &mut Scene,
    text: &mut SimpleText,
    viewport_width: f64,
    viewport_height: f64,
    profiles: &[GpuTimerQueryResult],
) {
    const COLORS: &[Color] = &[
        Color::AQUA,
        Color::RED,
        Color::ALICE_BLUE,
        Color::YELLOW,
        Color::GREEN,
        Color::BLUE,
        Color::ORANGE,
        Color::WHITE,
    ];
    if profiles_are_empty(profiles) {
        return;
    }
    let width = (viewport_width * 0.3).clamp(150., 450.);
    let height = width * 1.5;
    let y_offset = viewport_height - height;
    let offset = Affine::translate((0., y_offset));

    // Draw the background
    scene.fill(
        Fill::NonZero,
        offset,
        &Brush::Solid(Color::rgba8(0, 0, 0, 200)),
        None,
        &Rect::new(0., 0., width, height),
    );
    // Find the range of the samples, so we can normalise them
    let mut min = f64::MAX;
    let mut max = f64::MIN;
    let mut max_depth = 0;
    let mut depth = 0;
    let mut count = 0;
    traverse_profiling(profiles, &mut |profile, stage| {
        match stage {
            TraversalStage::Enter => {
                count += 1;
                if let Some(time) = &profile.time {
                    min = min.min(time.start);
                    max = max.max(time.end);
                }
                max_depth = max_depth.max(depth);
                // Apply a higher depth to the children
                depth += 1;
            }
            TraversalStage::Leave => depth -= 1,
        }
    });
    let total_time = max - min;
    {
        let labels = [
            format!("GPU Time: {:.2?}", Duration::from_secs_f64(total_time)),
            "Press P to save a trace".to_string(),
        ];

        // height / 5 is dedicated to the text labels and the rest is filled by the frame time.
        let text_height = height * 0.2 / (1 + labels.len()) as f64;
        let left_margin = width * 0.01;
        let text_size = (text_height * 0.9) as f32;
        for (i, label) in labels.iter().enumerate() {
            text.add(
                scene,
                None,
                text_size,
                Some(&Brush::Solid(Color::WHITE)),
                offset * Affine::translate((left_margin, (i + 1) as f64 * text_height)),
                label,
            );
        }

        let text_size = (text_height * 0.9) as f32;
        for (i, label) in labels.iter().enumerate() {
            text.add(
                scene,
                None,
                text_size,
                Some(&Brush::Solid(Color::WHITE)),
                offset * Affine::translate((left_margin, (i + 1) as f64 * text_height)),
                label,
            );
        }
    }
    let timeline_start_y = height * 0.21;
    let timeline_range_y = height * 0.78;
    let timeline_range_end = timeline_start_y + timeline_range_y;

    // Add 6 items worth of margin
    let text_height = timeline_range_y / (6 + count) as f64;
    let left_margin = width * 0.35;
    let mut cur_text_y = timeline_start_y;
    let mut cur_index = 0;
    let mut depth = 0;
    // Leave 1 bar's worth of margin
    let depth_width = width * 0.28 / (max_depth + 1) as f64;
    let depth_size = depth_width * 0.8;
    traverse_profiling(profiles, &mut |profile, stage| {
        if let TraversalStage::Enter = stage {
            if let Some(time) = &profile.time {
                let start_normalised =
                    ((time.start - min) / total_time) * timeline_range_y + timeline_start_y;
                let end_normalised =
                    ((time.end - min) / total_time) * timeline_range_y + timeline_start_y;

                let color = COLORS[cur_index % COLORS.len()];
                let x = width * 0.01 + (depth as f64 * depth_width);
                scene.fill(
                    Fill::NonZero,
                    offset,
                    &Brush::Solid(color),
                    None,
                    &Rect::new(x, start_normalised, x + depth_size, end_normalised),
                );

                let mut text_start = start_normalised;
                let nested = !profiles_are_empty(&profile.nested_queries);
                if nested {
                    // If we have children, leave some more space for them
                    text_start -= text_height * 0.7;
                }
                let this_time = time.end - time.start;
                // Highlight as important if more than 10% of the total time, or more than 1ms
                let slow = this_time * 20. >= total_time || this_time >= 0.001;
                let text_y = text_start
                    // Ensure that we don't overlap the previous item
                    .max(cur_text_y)
                    // Ensure that all remaining items can fit
                    .min(timeline_range_end - (count - cur_index) as f64 * text_height);
                let (text_height, text_color) = if slow {
                    (text_height, Color::WHITE)
                } else {
                    (text_height * 0.6, Color::LIGHT_GRAY)
                };
                let text_size = (text_height * 0.9) as f32;
                // Text is specified by the baseline, but the y positions all refer to the top of the text
                cur_text_y = text_y + text_height;
                let label = {
                    // Sometimes, the duration turns out to be negative
                    // We have not yet debugged this, but display the absolute value in that case
                    // see https://github.com/linebender/vello/pull/475 for more
                    if this_time < 0.0 {
                        format!(
                            "-{:.2?}(!!) - {:.30}",
                            Duration::from_secs_f64(this_time.abs()),
                            profile.label
                        )
                    } else {
                        format!(
                            "{:.2?} - {:.30}",
                            Duration::from_secs_f64(this_time),
                            profile.label
                        )
                    }
                };
                scene.fill(
                    Fill::NonZero,
                    offset,
                    &Brush::Solid(color),
                    None,
                    &Rect::new(
                        width * 0.31,
                        cur_text_y - text_size as f64 * 0.7,
                        width * 0.34,
                        cur_text_y,
                    ),
                );
                text.add(
                    scene,
                    None,
                    text_size,
                    Some(&Brush::Solid(text_color)),
                    offset * Affine::translate((left_margin, cur_text_y)),
                    &label,
                );
                if !nested && slow {
                    scene.stroke(
                        &Stroke::new(2.),
                        offset,
                        &Brush::Solid(color),
                        None,
                        &vello::kurbo::Line::new(
                            (x + depth_size, (end_normalised + start_normalised) / 2.),
                            (width * 0.31, cur_text_y - text_size as f64 * 0.35),
                        ),
                    );
                }
                cur_index += 1;
                // Higher depth applies only to the children
            }
            depth += 1;
        } else {
            depth -= 1;
        }
    });
}

enum TraversalStage {
    Enter,
    Leave,
}
fn traverse_profiling(
    profiles: &[GpuTimerQueryResult],
    callback: &mut impl FnMut(&GpuTimerQueryResult, TraversalStage),
) {
    for profile in profiles {
        callback(profile, TraversalStage::Enter);
        traverse_profiling(&profile.nested_queries, &mut *callback);
        callback(profile, TraversalStage::Leave);
    }
}

/// Simple hack to support loading images for examples.
#[derive(Default)]
pub struct ImageCache {
    files: HashMap<PathBuf, Image>,
    bytes: HashMap<usize, Image>,
}

impl ImageCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_file(&mut self, path: impl AsRef<Path>) -> anyhow::Result<Image> {
        let path = path.as_ref();
        if let Some(image) = self.files.get(path) {
            Ok(image.clone())
        } else {
            let data = std::fs::read(path)?;
            let image = decode_image(&data)?;
            self.files.insert(path.to_owned(), image.clone());
            Ok(image)
        }
    }

    pub fn from_bytes(&mut self, key: usize, bytes: &[u8]) -> anyhow::Result<Image> {
        if let Some(image) = self.bytes.get(&key) {
            Ok(image.clone())
        } else {
            let image = decode_image(bytes)?;
            self.bytes.insert(key, image.clone());
            Ok(image)
        }
    }
}

fn decode_image(data: &[u8]) -> anyhow::Result<Image> {
    let image = image::ImageReader::new(std::io::Cursor::new(data))
        .with_guessed_format()?
        .decode()?;
    let width = image.width();
    let height = image.height();
    let data = Arc::new(image.into_rgba8().into_vec());
    let blob = Blob::new(data);
    Ok(Image::new(blob, Format::Rgba8, width, height))
}

pub struct SceneParams<'a> {
    pub time: f64,
    /// Whether blocking should be limited
    /// Will not change between runs
    // TODO: Just never block/handle this automatically?
    pub interactive: bool,
    pub text: &'a mut SimpleText,
    pub images: &'a mut ImageCache,
    pub resolution: Option<Vec2>,
    pub base_color: Option<vello::peniko::Color>,
}

pub struct SceneConfig {
    // TODO: This is currently unused
    pub animated: bool,
    pub name: String,
}

pub struct ExampleScene {
    pub function: Box<dyn TestScene>,
    pub config: SceneConfig,
}

pub trait TestScene {
    fn render(&mut self, scene: &mut Scene, params: &mut SceneParams);
}

impl<F: FnMut(&mut Scene, &mut SceneParams)> TestScene for F {
    fn render(&mut self, scene: &mut Scene, params: &mut SceneParams) {
        self(scene, params);
    }
}

pub struct SceneSet {
    pub scenes: Vec<ExampleScene>,
}

const WIDTH: usize = 1600;
const HEIGHT: usize = 900;

const GRID_WIDTH: i64 = 80;
const GRID_HEIGHT: i64 = 40;

pub struct MMark {
    elements: Vec<Element>,
}

struct Element {
    seg: PathSeg,
    color: Color,
    width: f64,
    is_split: bool,
    grid_point: GridPoint,
}

#[derive(Clone, Copy)]
struct GridPoint(i64, i64);

impl MMark {
    pub fn new(n: usize) -> MMark {
        let mut result = MMark { elements: vec![] };
        result.resize(n);
        result
    }

    fn resize(&mut self, n: usize) {
        let old_n = self.elements.len();
        match n.cmp(&old_n) {
            Ordering::Less => self.elements.truncate(n),
            Ordering::Greater => {
                let mut last = self
                    .elements
                    .last()
                    .map(|e| e.grid_point)
                    .unwrap_or(GridPoint(GRID_WIDTH / 2, GRID_HEIGHT / 2));
                self.elements.extend((old_n..n).map(|_| {
                    let element = Element::new_rand(last);
                    last = element.grid_point;
                    element
                }));
            }
            _ => (),
        }
    }
}

impl TestScene for MMark {
    fn render(&mut self, scene: &mut Scene, _params: &mut SceneParams) {
        let mut rng = rand::thread_rng();
        let mut path = BezPath::new();
        let len = self.elements.len();
        for (i, element) in self.elements.iter_mut().enumerate() {
            if path.is_empty() {
                path.move_to(element.seg.start());
            }
            match element.seg {
                PathSeg::Line(l) => path.line_to(l.p1),
                PathSeg::Quad(q) => path.quad_to(q.p1, q.p2),
                PathSeg::Cubic(c) => path.curve_to(c.p1, c.p2, c.p3),
            }
            if element.is_split || i == len {
                // This gets color and width from the last element, original
                // gets it from the first, but this should not matter.
                scene.stroke(
                    &Stroke::new(element.width),
                    Affine::IDENTITY,
                    element.color,
                    None,
                    &path,
                );
                path.truncate(0); // Should have clear method, to avoid allocations.
            }
            if rng.gen::<f32>() > 0.995 {
                element.is_split ^= true;
            }
        }
    }
}

const COLORS: &[Color] = &[
    Color::rgb8(0x10, 0x10, 0x10),
    Color::rgb8(0x80, 0x80, 0x80),
    Color::rgb8(0xc0, 0xc0, 0xc0),
    Color::rgb8(0x10, 0x10, 0x10),
    Color::rgb8(0x80, 0x80, 0x80),
    Color::rgb8(0xc0, 0xc0, 0xc0),
    Color::rgb8(0xe0, 0x10, 0x40),
];

impl Element {
    fn new_rand(last: GridPoint) -> Element {
        let mut rng = rand::thread_rng();
        let seg_type = rng.gen_range(0..4);
        let next = GridPoint::random_point(last);
        let (grid_point, seg) = if seg_type < 2 {
            (
                next,
                PathSeg::Line(Line::new(last.coordinate(), next.coordinate())),
            )
        } else if seg_type < 3 {
            let p2 = GridPoint::random_point(next);
            (
                p2,
                PathSeg::Quad(QuadBez::new(
                    last.coordinate(),
                    next.coordinate(),
                    p2.coordinate(),
                )),
            )
        } else {
            let p2 = GridPoint::random_point(next);
            let p3 = GridPoint::random_point(next);
            (
                p3,
                PathSeg::Cubic(CubicBez::new(
                    last.coordinate(),
                    next.coordinate(),
                    p2.coordinate(),
                    p3.coordinate(),
                )),
            )
        };
        let color = *COLORS.choose(&mut rng).unwrap();
        let width = rng.gen::<f64>().powi(5) * 20.0 + 1.0;
        let is_split = rng.gen();
        Element {
            seg,
            color,
            width,
            is_split,
            grid_point,
        }
    }
}

const OFFSETS: &[(i64, i64)] = &[(-4, 0), (2, 0), (1, -2), (1, 2)];

impl GridPoint {
    fn random_point(last: GridPoint) -> GridPoint {
        let mut rng = rand::thread_rng();

        let offset = OFFSETS.choose(&mut rng).unwrap();
        let mut x = last.0 + offset.0;
        if !(0..=GRID_WIDTH).contains(&x) {
            x -= offset.0 * 2;
        }
        let mut y = last.1 + offset.1;
        if !(0..=GRID_HEIGHT).contains(&y) {
            y -= offset.1 * 2;
        }
        GridPoint(x, y)
    }

    fn coordinate(&self) -> Point {
        let scale_x = WIDTH as f64 / ((GRID_WIDTH + 1) as f64);
        let scale_y = HEIGHT as f64 / ((GRID_HEIGHT + 1) as f64);
        Point::new(
            (self.0 as f64 + 0.5) * scale_x,
            100.0 + (self.1 as f64 + 0.5) * scale_y,
        )
    }
}

pub struct PicoSvg {
    pub items: Vec<Item>,
    pub size: Size,
}

pub enum Item {
    Fill(FillItem),
    Stroke(StrokeItem),
    Group(GroupItem),
}

pub struct StrokeItem {
    pub width: f64,
    pub color: Color,
    pub path: BezPath,
}

pub struct FillItem {
    pub color: Color,
    pub path: BezPath,
}

pub struct GroupItem {
    pub affine: Affine,
    pub children: Vec<Item>,
}

struct Parser {
    scale: f64,
}

impl PicoSvg {
    pub fn load(xml_string: &str, scale: f64) -> Result<PicoSvg, Box<dyn std::error::Error>> {
        let doc = Document::parse(xml_string)?;
        let root = doc.root_element();
        let mut parser = Parser::new(scale);
        let width = root.attribute("width").and_then(|s| f64::from_str(s).ok());
        let height = root.attribute("height").and_then(|s| f64::from_str(s).ok());
        let (origin, viewbox_size) = root
            .attribute("viewBox")
            .and_then(|vb_attr| {
                let vs: Vec<f64> = vb_attr
                    .split(' ')
                    .map(|s| f64::from_str(s).unwrap())
                    .collect();
                if let &[x, y, width, height] = vs.as_slice() {
                    Some((Point { x, y }, Size { width, height }))
                } else {
                    None
                }
            })
            .unzip();

        let mut transform = if let Some(origin) = origin {
            Affine::translate(origin.to_vec2() * -1.0)
        } else {
            Affine::IDENTITY
        };

        transform *= match (width, height, viewbox_size) {
            (None, None, Some(_)) => Affine::IDENTITY,
            (Some(w), Some(h), Some(s)) => {
                Affine::scale_non_uniform(1.0 / s.width * w, 1.0 / s.height * h)
            }
            (Some(w), None, Some(s)) => Affine::scale(1.0 / s.width * w),
            (None, Some(h), Some(s)) => Affine::scale(1.0 / s.height * h),
            _ => Affine::IDENTITY,
        };

        let size = match (width, height, viewbox_size) {
            (None, None, Some(s)) => s,
            (mw, mh, None) => Size {
                width: mw.unwrap_or(300_f64),
                height: mh.unwrap_or(150_f64),
            },
            (Some(w), None, Some(s)) => Size {
                width: w,
                height: 1.0 / w * s.width * s.height,
            },
            (None, Some(h), Some(s)) => Size {
                width: 1.0 / h * s.height * s.width,
                height: h,
            },
            (Some(width), Some(height), Some(_)) => Size { width, height },
        };

        transform *= if scale >= 0.0 {
            Affine::scale(scale)
        } else {
            Affine::new([-scale, 0.0, 0.0, scale, 0.0, 0.0])
        };
        let props = RecursiveProperties {
            fill: Some(Color::BLACK),
        };
        // The root element is the svg document element, which we don't care about
        let mut items = Vec::new();
        for node in root.children() {
            parser.rec_parse(node, &props, &mut items)?;
        }
        let root_group = Item::Group(GroupItem {
            affine: transform,
            children: items,
        });
        Ok(PicoSvg {
            items: vec![root_group],
            size,
        })
    }
}

#[derive(Clone)]
struct RecursiveProperties {
    fill: Option<Color>,
}

impl Parser {
    fn new(scale: f64) -> Parser {
        Parser { scale }
    }

    fn rec_parse(
        &mut self,
        node: Node,
        properties: &RecursiveProperties,
        items: &mut Vec<Item>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if node.is_element() {
            let mut properties = properties.clone();
            if let Some(fill_color) = node.attribute("fill") {
                if fill_color == "none" {
                    properties.fill = None;
                } else {
                    let color = parse_color(fill_color);
                    let color = modify_opacity(color, "fill-opacity", node);
                    // TODO: Handle recursive opacity properly
                    let color = modify_opacity(color, "opacity", node);
                    properties.fill = Some(color);
                }
            }
            match node.tag_name().name() {
                "g" => {
                    let mut children = Vec::new();
                    let mut affine = Affine::default();
                    if let Some(transform) = node.attribute("transform") {
                        affine = parse_transform(transform);
                    }
                    for child in node.children() {
                        self.rec_parse(child, &properties, &mut children)?;
                    }
                    items.push(Item::Group(GroupItem { affine, children }));
                }
                "path" => {
                    let d = node.attribute("d").ok_or("missing 'd' attribute")?;
                    let bp = BezPath::from_svg(d)?;
                    let path = bp;
                    if let Some(color) = properties.fill {
                        items.push(Item::Fill(FillItem {
                            color,
                            path: path.clone(),
                        }));
                    }
                    if let Some(stroke_color) = node.attribute("stroke") {
                        if stroke_color != "none" {
                            let width = node
                                .attribute("stroke-width")
                                .map(|a| f64::from_str(a).unwrap_or(1.0))
                                .unwrap_or(1.0)
                                * self.scale.abs();
                            let color = parse_color(stroke_color);
                            let color = modify_opacity(color, "stroke-opacity", node);
                            // TODO: Handle recursive opacity properly
                            let color = modify_opacity(color, "opacity", node);
                            items.push(Item::Stroke(StrokeItem { width, color, path }));
                        }
                    }
                }
                other => eprintln!("Unhandled node type {other}"),
            }
        }
        Ok(())
    }
}

fn parse_transform(transform: &str) -> Affine {
    let mut nt = Affine::IDENTITY;
    for ts in transform.split(')').map(str::trim) {
        nt *= if let Some(s) = ts.strip_prefix("matrix(") {
            let vals = s
                .split([',', ' '])
                .map(str::parse)
                .collect::<Result<Vec<f64>, _>>()
                .expect("Could parse all values of 'matrix' as floats");
            Affine::new(
                vals.try_into()
                    .expect("Should be six arguments to `matrix`"),
            )
        } else if let Some(s) = ts.strip_prefix("translate(") {
            if let Ok(vals) = s
                .split([',', ' '])
                .map(str::trim)
                .map(str::parse)
                .collect::<Result<Vec<f64>, _>>()
            {
                match vals.as_slice() {
                    &[x, y] => Affine::translate(Vec2 { x, y }),
                    _ => Affine::IDENTITY,
                }
            } else {
                Affine::IDENTITY
            }
        } else if let Some(s) = ts.strip_prefix("scale(") {
            if let Ok(vals) = s
                .split([',', ' '])
                .map(str::trim)
                .map(str::parse)
                .collect::<Result<Vec<f64>, _>>()
            {
                match *vals.as_slice() {
                    [x, y] => Affine::scale_non_uniform(x, y),
                    [x] => Affine::scale(x),
                    _ => Affine::IDENTITY,
                }
            } else {
                Affine::IDENTITY
            }
        } else if let Some(s) = ts.strip_prefix("scaleX(") {
            s.trim()
                .parse()
                .ok()
                .map(|x| Affine::scale_non_uniform(x, 1.0))
                .unwrap_or(Affine::IDENTITY)
        } else if let Some(s) = ts.strip_prefix("scaleY(") {
            s.trim()
                .parse()
                .ok()
                .map(|y| Affine::scale_non_uniform(1.0, y))
                .unwrap_or(Affine::IDENTITY)
        } else {
            if !ts.is_empty() {
                eprintln!("Did not understand transform attribute {ts:?})");
            }
            Affine::IDENTITY
        };
    }
    nt
}

fn parse_color(color: &str) -> Color {
    let color = color.trim();
    if let Some(c) = Color::parse(color) {
        c
    } else if let Some(s) = color.strip_prefix("rgb(").and_then(|s| s.strip_suffix(')')) {
        let mut iter = s.split([',', ' ']).map(str::trim).map(u8::from_str);

        let r = iter.next().unwrap().unwrap();
        let g = iter.next().unwrap().unwrap();
        let b = iter.next().unwrap().unwrap();
        Color::rgb8(r, g, b)
    } else {
        Color::rgba8(255, 0, 255, 0x80)
    }
}

fn modify_opacity(mut color: Color, attr_name: &str, node: Node) -> Color {
    if let Some(opacity) = node.attribute(attr_name) {
        let alpha: f64 = if let Some(o) = opacity.strip_suffix('%') {
            let pctg = o.parse().unwrap_or(100.0);
            pctg * 0.01
        } else {
            opacity.parse().unwrap_or(1.0)
        };
        color.a = (alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
        color
    } else {
        color
    }
}

// This is very much a hack to get things working.
// On Windows, can set this to "c:\\Windows\\Fonts\\seguiemj.ttf" to get color emoji
const ROBOTO_FONT: &[u8] = include_bytes!("../assets/roboto/Roboto-Regular.ttf");
const INCONSOLATA_FONT: &[u8] = include_bytes!("../assets/inconsolata/Inconsolata.ttf");
const NOTO_EMOJI_CBTF_SUBSET: &[u8] =
    include_bytes!("../assets/noto_color_emoji/NotoColorEmoji-CBTF-Subset.ttf");
const NOTO_EMOJI_COLR_SUBSET: &[u8] =
    include_bytes!("../assets/noto_color_emoji/NotoColorEmoji-Subset.ttf");

pub struct SimpleText {
    roboto: Font,
    inconsolata: Font,
    noto_emoji_colr_subset: Font,
    noto_emoji_cbtf_subset: Font,
}

impl SimpleText {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            roboto: Font::new(Blob::new(Arc::new(ROBOTO_FONT)), 0),
            inconsolata: Font::new(Blob::new(Arc::new(INCONSOLATA_FONT)), 0),
            noto_emoji_colr_subset: Font::new(Blob::new(Arc::new(NOTO_EMOJI_COLR_SUBSET)), 0),
            noto_emoji_cbtf_subset: Font::new(Blob::new(Arc::new(NOTO_EMOJI_CBTF_SUBSET)), 0),
        }
    }

    /// Add a text run which supports some emoji.
    ///
    /// The supported Emoji are ✅, 👀, 🎉, and 🤠.
    /// This subset is chosen to demonstrate the emoji support, whilst
    /// not significantly increasing repository size.
    ///
    /// Note that Vello does support COLR emoji, but does not currently support
    /// any other forms of emoji.
    #[allow(clippy::too_many_arguments)]
    pub fn add_colr_emoji_run<'a>(
        &mut self,
        scene: &mut Scene,
        size: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        style: impl Into<StyleRef<'a>>,
        text: &str,
    ) {
        let font = self.noto_emoji_colr_subset.clone();
        self.add_var_run(
            scene,
            Some(&font),
            size,
            &[],
            // This should be unused
            &Brush::Solid(Color::WHITE),
            transform,
            glyph_transform,
            style,
            text,
        );
    }

    /// Add a text run which supports some emoji.
    ///
    /// The supported Emoji are ✅, 👀, 🎉, and 🤠.
    /// This subset is chosen to demonstrate the emoji support, whilst
    /// not significantly increasing repository size.
    ///
    /// This will use a CBTF font, which Vello supports.
    #[allow(clippy::too_many_arguments)]
    pub fn add_bitmap_emoji_run<'a>(
        &mut self,
        scene: &mut Scene,
        size: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        style: impl Into<StyleRef<'a>>,
        text: &str,
    ) {
        let font = self.noto_emoji_cbtf_subset.clone();
        self.add_var_run(
            scene,
            Some(&font),
            size,
            &[],
            // This should be unused
            &Brush::Solid(Color::WHITE),
            transform,
            glyph_transform,
            style,
            text,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_run<'a>(
        &mut self,
        scene: &mut Scene,
        font: Option<&Font>,
        size: f32,
        brush: impl Into<BrushRef<'a>>,
        transform: Affine,
        glyph_transform: Option<Affine>,
        style: impl Into<StyleRef<'a>>,
        text: &str,
    ) {
        self.add_var_run(
            scene,
            font,
            size,
            &[],
            brush,
            transform,
            glyph_transform,
            style,
            text,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_var_run<'a>(
        &mut self,
        scene: &mut Scene,
        font: Option<&Font>,
        size: f32,
        variations: &[(&str, f32)],
        brush: impl Into<BrushRef<'a>>,
        transform: Affine,
        glyph_transform: Option<Affine>,
        style: impl Into<StyleRef<'a>>,
        text: &str,
    ) {
        let default_font = if variations.is_empty() {
            &self.roboto
        } else {
            &self.inconsolata
        };
        let font = font.unwrap_or(default_font);
        let font_ref = to_font_ref(font).unwrap();
        let brush = brush.into();
        let style = style.into();
        let axes = font_ref.axes();
        let font_size = vello::skrifa::instance::Size::new(size);
        let var_loc = axes.location(variations.iter().copied());
        let charmap = font_ref.charmap();
        let metrics = font_ref.metrics(font_size, &var_loc);
        let line_height = metrics.ascent - metrics.descent + metrics.leading;
        let glyph_metrics = font_ref.glyph_metrics(font_size, &var_loc);
        let mut pen_x = 0f32;
        let mut pen_y = 0f32;
        scene
            .draw_glyphs(font)
            .font_size(size)
            .transform(transform)
            .glyph_transform(glyph_transform)
            .normalized_coords(var_loc.coords())
            .brush(brush)
            .hint(false)
            .draw(
                style,
                text.chars().filter_map(|ch| {
                    if ch == '\n' {
                        pen_y += line_height;
                        pen_x = 0.0;
                        return None;
                    }
                    let gid = charmap.map(ch).unwrap_or_default();
                    let advance = glyph_metrics.advance_width(gid).unwrap_or_default();
                    let x = pen_x;
                    pen_x += advance;
                    Some(Glyph {
                        id: gid.to_u32(),
                        x,
                        y: pen_y,
                    })
                }),
            );
    }

    pub fn add(
        &mut self,
        scene: &mut Scene,
        font: Option<&Font>,
        size: f32,
        brush: Option<&Brush>,
        transform: Affine,
        text: &str,
    ) {
        use vello::peniko::Fill;
        let brush = brush.unwrap_or(&Brush::Solid(Color::WHITE));
        self.add_run(
            scene,
            font,
            size,
            brush,
            transform,
            None,
            Fill::NonZero,
            text,
        );
    }
}

fn to_font_ref(font: &Font) -> Option<FontRef<'_>> {
    use vello::skrifa::raw::FileRef;
    let file_ref = FileRef::new(font.data.as_ref()).ok()?;
    match file_ref {
        FileRef::Font(font) => Some(font),
        FileRef::Collection(collection) => collection.get(font.index).ok(),
    }
}

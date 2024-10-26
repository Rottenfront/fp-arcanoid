use core::f64;
use std::time::Instant;

use vello::{
    kurbo::{Affine, Circle, Line, Point, Rect, Size, Stroke, Vec2},
    peniko::{Brush, Color, Fill},
    Scene,
};

use crate::SimpleText;

#[derive(Clone, Copy)]
pub struct Ball {
    pub circle: Circle,
    pub velocity: Vec2,
}

#[derive(Clone, Copy)]
pub struct Caret {
    pub pos: Point,
    pub velocity: Vec2,
    pub size: Size,
}

#[derive(Clone, Copy)]
pub struct Stone {
    pub rect: Rect,
    pub disposable: bool,
    pub disposed: bool,
}

#[derive(Clone)]
pub struct GameState {
    pub balls: Vec<Ball>,
    pub stones: Vec<Stone>,
    pub caret: Caret,
    pub rect: Rect,
    pub loss: bool,
    pub count: i32,
}

impl GameState {
    fn new() -> Self {
        let stone_size = (90., 30.);
        let stones = (1..14)
            .map(|i| {
                (1..5)
                    .map(|j| Stone {
                        rect: Rect::from_origin_size((100. * i as f64, 50. * j as f64), stone_size),
                        disposable: true,
                        disposed: false,
                    })
                    .collect::<Vec<Stone>>()
            })
            .collect::<Vec<Vec<Stone>>>()
            .concat();
        Self {
            balls: vec![Ball {
                circle: Circle::new((400., 300.), 10.),
                velocity: (0.5, 0.5).into(),
            }],
            stones,
            caret: Caret {
                pos: (800., 800.).into(),
                velocity: (0.7, 0.).into(),
                size: (200., 30.).into(),
            },
            rect: Rect::new(10., 10., 1600., 900.),
            loss: false,
            count: 0,
        }
    }
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy)]
pub enum CaretDirection {
    None,
    Left,
    Right,
}

#[derive(Clone, Copy)]
pub struct Changes {
    pub prev_instant: Instant,
    pub now: Instant,
    pub caret_direction: CaretDirection,
    pub restart: bool,
}

fn move_caret(mut state: GameState, duration: u128, direction: CaretDirection) -> GameState {
    match direction {
        CaretDirection::None => {}
        CaretDirection::Left => {
            state.caret.pos -= state.caret.velocity * duration as f64;
        }
        CaretDirection::Right => {
            state.caret.pos += state.caret.velocity * duration as f64;
        }
    }
    state
}

fn check_border_collision(mut ball: Ball, border: Rect) -> (Ball, bool) {
    let hitbox = Rect::from_center_size(
        ball.circle.center,
        Size::new(ball.circle.radius * 2., ball.circle.radius * 2.),
    );

    let mut loss = false;

    if hitbox.x0 <= border.x0 {
        ball.velocity.x = -ball.velocity.x;
        ball.circle.center.x += (border.x0 - hitbox.x0) * 3.;
    }
    if hitbox.x1 >= border.x1 {
        ball.velocity.x = -ball.velocity.x;
        ball.circle.center.x += (border.x1 - hitbox.x1) * 3.;
    }
    if hitbox.y0 <= border.y0 {
        ball.velocity.y = -ball.velocity.y;
        ball.circle.center.y += (border.y0 - hitbox.y0) * 3.;
    }
    if hitbox.y1 >= border.y1 {
        ball.velocity.y = -ball.velocity.y;
        ball.circle.center.y += (border.y1 - hitbox.y1) * 3.;
        loss = true;
    }

    (ball, loss)
}

fn get_line_point_dist(line: Line, point: Point) -> f64 {
    let Line {
        p0: Point { x: x0, y: y0 },
        p1: Point { x: x1, y: y1 },
    } = line;
    let a = y0 - y1;
    let b = x1 - x0;
    let c = x0 * y1 - x1 * y0;
    (a * point.x + b * point.y + c).abs() / (a * a + b * b).sqrt()
}

fn collides_circle_line(circle: Circle, line: Line) -> bool {
    (circle.center - line.p0).hypot().abs() <= circle.radius
        || (circle.center - line.p1).hypot().abs() <= circle.radius
        || {
            let Line {
                p0: Point { x: x0, y: y0 },
                p1: Point { x: x1, y: y1 },
            } = line.clone();
            if x0 == x1 {
                let center = circle.center;
                let dist = get_line_point_dist(line, center);
                dist <= circle.radius && center.y < y0.max(y1) && center.y > y0.min(y1)
            } else if y0 == y1 {
                let center = circle.center;
                let dist = get_line_point_dist(line, center);
                dist <= circle.radius && center.x < x0.max(x1) && center.x > x0.min(x1)
            } else {
                let vec = Vec2::new(x1 - x0, y1 - x0).normalize();
                let angle = f64::consts::PI / 2.;
                let normal = Vec2::new(
                    vec.x * angle.cos() - vec.y * angle.sin(),
                    vec.x * angle.sin() + vec.y * angle.cos(),
                );
                let p0 = Point::new(x0, y0);
                let p1 = Point::new(x1, y1);
                let is_in_range = normal.cross(circle.center - p0) <= 0.
                    && normal.cross(circle.center - p1) >= 0.;

                is_in_range && get_line_point_dist(line, circle.center) <= circle.radius
            }
        }
}

fn check_stone_collision(mut ball: Ball, mut stone: Stone) -> (Ball, Stone) {
    let circle = ball.circle;
    let rect = stone.rect;

    let p0 = rect.origin();
    let p1 = Point {
        x: p0.x + rect.width(),
        y: p0.y,
    };
    let p2 = Point {
        x: p1.x,
        y: p1.y + rect.height(),
    };
    let p3 = Point {
        x: p0.x,
        y: p0.y + rect.height(),
    };

    if collides_circle_line(circle, Line::new(p0, p1)) {
        ball.velocity.y = -ball.velocity.y;
        ball.circle.center.y += (p0.y - (circle.center.y + circle.radius)) * 1.;
        if stone.disposable {
            stone.disposed = true;
        }
    } else if collides_circle_line(circle, Line::new(p2, p3)) {
        ball.velocity.y = -ball.velocity.y;
        ball.circle.center.y -= (p2.y - (circle.center.y + circle.radius)) * 1.;
        if stone.disposable {
            stone.disposed = true;
        }
    } else if collides_circle_line(circle, Line::new(p0, p3)) {
        ball.velocity.x = -ball.velocity.x;
        ball.circle.center.x += (p0.x - (circle.center.x + circle.radius)) * 1.;
        if stone.disposable {
            stone.disposed = true;
        }
    } else if collides_circle_line(circle, Line::new(p1, p2)) {
        ball.velocity.x = -ball.velocity.x;
        ball.circle.center.x -= (p1.x - (circle.center.x + circle.radius)) * 1.;
        if stone.disposable {
            stone.disposed = true;
        }
    }

    (ball, stone)
}

fn move_balls(mut state: GameState, duration: u128) -> GameState {
    let balls = std::mem::take(&mut state.balls);
    let mut stones = std::mem::take(&mut state.stones);
    state.balls = balls
        .iter()
        .map(|ball| {
            let Ball { circle, velocity } = ball.clone();
            let circle = Circle::new(circle.center + velocity * duration as f64, circle.radius);
            let (mut ball, loss) = check_border_collision(Ball { circle, velocity }, state.rect);
            state.loss |= loss;

            ball = check_stone_collision(
                ball,
                Stone {
                    rect: Rect::from_center_size(state.caret.pos, state.caret.size),
                    disposable: false,
                    disposed: false,
                },
            )
            .0;

            let prev_stone_count = stones.len();

            stones = stones
                .iter()
                .map(|stone| {
                    let (new_ball, stone) = check_stone_collision(ball, *stone);
                    ball = new_ball;
                    stone
                })
                .filter(|stone| !stone.disposed)
                .collect();

            state.count += (prev_stone_count - stones.len()) as i32;

            ball
        })
        .collect();
    state.stones = stones;
    state
}

fn modify_state(state: GameState, changes: Changes) -> GameState {
    if changes.restart {
        GameState::new()
    } else if !state.loss {
        let duration = (changes.now - changes.prev_instant).as_millis();
        let state = move_caret(state, duration, changes.caret_direction);
        let state = move_balls(state, duration);
        state
    } else {
        state
    }
}

fn draw_stones(state: GameState, mut scene: Scene) -> (GameState, Scene) {
    let color = Color::BLUE;
    let border_color = Color::BLACK;
    let stroke = Stroke::new(1.);
    state.stones.iter().for_each(|stone| {
        scene.fill(Fill::NonZero, Affine::IDENTITY, color, None, &stone.rect);
        scene.stroke(&stroke, Affine::IDENTITY, border_color, None, &stone.rect);
    });
    (state, scene)
}

fn draw_balls(state: GameState, mut scene: Scene) -> (GameState, Scene) {
    let color = Color::RED;
    let border_color = Color::BLACK;
    let stroke = Stroke::new(2.);
    state.balls.iter().for_each(|ball| {
        scene.fill(Fill::NonZero, Affine::IDENTITY, color, None, &ball.circle);
        scene.stroke(&stroke, Affine::IDENTITY, border_color, None, &ball.circle);
    });
    (state, scene)
}

fn draw_caret(state: GameState, mut scene: Scene) -> (GameState, Scene) {
    let color = Color::RED;
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        color,
        None,
        &Rect::from_center_size(state.caret.pos, state.caret.size),
    );
    (state, scene)
}

fn draw_borders(state: GameState, mut scene: Scene) -> (GameState, Scene) {
    let stroke = Stroke::new(2.);
    let color = Color::BLACK;
    scene.stroke(&stroke, Affine::IDENTITY, color, None, &state.rect);
    (state, scene)
}

fn draw_count(
    state: GameState,
    mut scene: Scene,
    mut text: SimpleText,
) -> (GameState, Scene, SimpleText) {
    text.add(
        &mut scene,
        None,
        20.,
        Some(&Brush::Solid(Color::BLACK)),
        Affine::translate((20., 40.)),
        &format!("Count: {}", state.count),
    );

    (state, scene, text)
}

fn draw_loss(
    state: GameState,
    mut scene: Scene,
    mut text: SimpleText,
) -> (GameState, Scene, SimpleText) {
    if state.loss {
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            Color::rgba(0.0, 0.0, 0.0, 0.5),
            None,
            &state.rect,
        );
        text.add(
            &mut scene,
            None,
            40.,
            Some(&Brush::Solid(Color::RED)),
            Affine::translate((700., 450.)),
            &format!("GAME ENDED"),
        );
    }

    (state, scene, text)
}

fn render(state: GameState) -> (GameState, Scene) {
    let (state, scene) = draw_balls(state, Scene::new());
    let (state, scene) = draw_caret(state, scene);
    let (state, scene) = draw_stones(state, scene);
    let (state, scene) = draw_borders(state, scene);
    let text = SimpleText::new();
    let (state, scene, text) = draw_loss(state, scene, text);
    let (state, scene, text) = draw_count(state, scene, text);

    (state, scene)
}

pub fn run_game(state: GameState, changes: Changes) -> (GameState, Scene) {
    let state = modify_state(state, changes);
    render(state)
}

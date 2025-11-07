"""Pygame-based renderer for the Flappy RL environment."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pygame

from .physics import BIRD_SIZE, HEIGHT, PIPE_W, WIDTH


class Renderer:
    def __init__(self, width: int = WIDTH, height: int = HEIGHT, caption: str = "Flappy RL") -> None:
        pygame.init()
        pygame.display.set_caption(caption)
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height
        self.font: Optional[pygame.font.Font] = None
        self.cloud_offset = 0.0

    def draw(self, state: dict) -> np.ndarray:
        if self.font is None:
            self.font = pygame.font.SysFont("Arial", 18)

        pygame.event.pump()
        wind = state.get("wind", 0.0)
        self._draw_background(wind)

        # Pipes
        top_rect, bottom_rect = state["pipe_rects"]
        self._draw_pipe(top_rect)
        self._draw_pipe(bottom_rect)

        # Bird
        bird_pos = state["bird_pos"]
        bird_angle = state.get("bird_angle", 0.0)
        wing_phase = state.get("wing_phase", 0.0)
        self._draw_bird(bird_pos, bird_angle, wing_phase)

        # HUD
        text = f"pipes: {state['pipes_passed']}  steps: {state['t']}"
        if self.font:
            surface = self.font.render(text, True, (10, 10, 10))
            self.screen.blit(surface, (10, 10))
        self._draw_wind_indicator(wind)

        pygame.display.flip()
        self.clock.tick(60)
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def close(self) -> None:
        pygame.display.quit()
        pygame.quit()

    def _draw_background(self, wind: float) -> None:
        sky_top = (120, 190, 255)
        sky_bottom = (80, 160, 230)
        for y in range(self.height):
            ratio = y / self.height
            color = tuple(
                int(sky_top[i] * (1 - ratio) + sky_bottom[i] * ratio) for i in range(3)
            )
            self.screen.fill(color, rect=pygame.Rect(0, y, self.width, 1))
        ground_rect = pygame.Rect(0, self.height - 60, self.width, 60)
        pygame.draw.rect(self.screen, (222, 216, 149), ground_rect)
        pygame.draw.rect(self.screen, (205, 200, 132), ground_rect.move(0, 6))

        self.cloud_offset = (self.cloud_offset + wind * 5) % (self.width * 2)
        for i in range(-1, 3):
            cx = (i * 160 + self.cloud_offset) % (self.width + 160) - 80
            cy = 80 + (i % 2) * 20
            self._draw_cloud(cx, cy)

    def _draw_pipe(self, rect: Tuple[float, float, float, float]) -> None:
        x, y, w, h = rect
        if h <= 0:
            return
        body_color = (34, 177, 76)
        lip_color = (50, 125, 50)
        main_rect = pygame.Rect(int(x), int(y), int(w), int(h))
        pygame.draw.rect(self.screen, body_color, main_rect)
        pygame.draw.rect(
            self.screen,
            lip_color,
            pygame.Rect(
                int(x - 4),
                int(y - 6 if y == 0 else y + h - 6),
                int(w + 8),
                12,
            ),
        )

    def _draw_bird(self, pos: Tuple[float, float], angle: float, wing_phase: float) -> None:
        bird_x, bird_y = pos

        # Shadow
        shadow = pygame.Surface((BIRD_SIZE * 2, BIRD_SIZE), pygame.SRCALPHA)
        pygame.draw.ellipse(
            shadow,
            (0, 0, 0, 70),
            shadow.get_rect().inflate(-10, 5),
        )
        shadow_rect = shadow.get_rect(center=(bird_x + 6, bird_y + 12))
        self.screen.blit(shadow, shadow_rect)

        body = pygame.Surface((BIRD_SIZE * 2, BIRD_SIZE * 2), pygame.SRCALPHA)
        body_rect = body.get_rect()
        center = body_rect.center

        pygame.draw.ellipse(body, (255, 225, 80), body_rect.inflate(-6, -2))
        pygame.draw.ellipse(body, (255, 255, 255, 200), body_rect.inflate(-10, -8))

        wing_lift = int(wing_phase * 8)
        wing_points = [
            (center[0] - 6, center[1] + 2),
            (center[0] + 8, center[1] - wing_lift),
            (center[0] + 12, center[1] + 6),
        ]
        pygame.draw.polygon(body, (240, 190, 40), wing_points)
        pygame.draw.lines(body, (180, 140, 30), False, wing_points, 2)

        beak = [
            (center[0] + 18, center[1] - 2),
            (center[0] + 28, center[1]),
            (center[0] + 18, center[1] + 2),
        ]
        pygame.draw.polygon(body, (255, 160, 20), beak)

        eye_center = (center[0] + 6, center[1] - 6)
        pygame.draw.circle(body, (255, 255, 255), eye_center, 5)
        pygame.draw.circle(body, (0, 0, 0), eye_center, 2)

        rotated = pygame.transform.rotate(body, angle)
        rotated_rect = rotated.get_rect(center=(bird_x, bird_y))
        self.screen.blit(rotated, rotated_rect)

    def _draw_cloud(self, x: float, y: float) -> None:
        cloud = pygame.Surface((120, 60), pygame.SRCALPHA)
        pygame.draw.ellipse(cloud, (255, 255, 255, 210), cloud.get_rect().inflate(-40, -20))
        cloud_rect = cloud.get_rect(center=(int(x), int(y)))
        self.screen.blit(cloud, cloud_rect)

    def _draw_wind_indicator(self, wind: float) -> None:
        bar_width = 120
        bar_height = 12
        center_x = self.width - 150
        center_y = 30
        bg_rect = pygame.Rect(center_x - bar_width // 2, center_y - bar_height // 2, bar_width, bar_height)
        pygame.draw.rect(self.screen, (240, 240, 240, 180), bg_rect, border_radius=6)
        magnitude = max(-1.0, min(1.0, wind))
        fill_width = int((bar_width / 2) * abs(magnitude))
        if magnitude > 0:
            fill_rect = pygame.Rect(center_x, center_y - bar_height // 2, fill_width, bar_height)
            color = (255, 120, 80)
        else:
            fill_rect = pygame.Rect(center_x - fill_width, center_y - bar_height // 2, fill_width, bar_height)
            color = (80, 160, 255)
        pygame.draw.rect(self.screen, color, fill_rect, border_radius=6)
        pygame.draw.line(self.screen, (60, 60, 60), (center_x, center_y - 8), (center_x, center_y + 8), 2)
        arrow_dir = 1 if magnitude >= 0 else -1
        arrow_length = 20 + 10 * abs(magnitude)
        start = (center_x, center_y + 20)
        end = (center_x + arrow_dir * arrow_length, center_y + 20)
        pygame.draw.line(self.screen, color, start, end, 3)
        head = arrow_dir * 6
        pygame.draw.polygon(
            self.screen,
            color,
            [
                (end[0], end[1]),
                (end[0] - head, end[1] - 4),
                (end[0] - head, end[1] + 4),
            ],
        )

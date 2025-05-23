"""Ijai map image parser."""

import logging

from PIL import Image
from PIL.Image import Image as ImageType
from PIL.Image import Resampling

from custom_components.xiaomi_cloud_map_extractor.common.image_handler import ImageHandler
from custom_components.xiaomi_cloud_map_extractor.const import *
from custom_components.xiaomi_cloud_map_extractor.types import Colors, ImageConfig

_LOGGER = logging.getLogger(__name__)

class ImageHandlerIjai(ImageHandler):
    MAP_OUTSIDE = 0x00
    MAP_WALL = 0xFF
    MAP_SCAN = 0x01
    MAP_NEW_DISCOVERED_AREA = 0x02
    MAP_ROOM_MIN = 10
    MAP_ROOM_MAX = 59
    MAP_SELECTED_ROOM_MIN = 60
    MAP_SELECTED_ROOM_MAX = 109

    @staticmethod
    def parse(map_data: bytes, width: int, height: int, colors: Colors, image_config: ImageConfig, draw_cleaned_area: bool
    ) -> tuple[ImageType | None, dict[int, tuple[int, int, int, int]], set[int], ImageType | None]:
        rooms = {}
        cleaned_areas = set()
        _LOGGER.debug(f"ijai parser: image_config = {image_config}")
        scale = image_config[CONF_SCALE]
        trim_left = int(image_config[CONF_TRIM][CONF_LEFT] * width / 100)
        trim_right = int(image_config[CONF_TRIM][CONF_RIGHT] * width / 100)
        trim_top = int(image_config[CONF_TRIM][CONF_TOP] * height / 100)
        trim_bottom = int(image_config[CONF_TRIM][CONF_BOTTOM] * height / 100)
        trimmed_height = height - trim_top - trim_bottom
        trimmed_width = width - trim_left - trim_right
        if trimmed_width == 0 or trimmed_height == 0:
            return ImageHandler.create_empty_map_image(colors), rooms, cleaned_areas, None

        image = Image.new('RGBA', (trimmed_width, trimmed_height))
        pixels = image.load()
        cleaned_areas_layer = None
        cleaned_areas_pixels = None
        if draw_cleaned_area:
            cleaned_areas_layer = Image.new("RGBA", (trimmed_width, trimmed_height))
            cleaned_areas_pixels = cleaned_areas_layer.load()
        _LOGGER.debug(f"trim_bottom = {trim_bottom}, trim_top = {trim_top}, trim_left = {trim_left}, trim_right = {trim_right}")
        unknown_pixels = set()
        for img_y in range(trimmed_height):
            y = trimmed_height - 1 - img_y
            for img_x in range(trimmed_width):
                x = img_x
                pixel_type = map_data[(img_y + trim_bottom)*width + x + trim_left]
                if pixel_type == ImageHandlerIjai.MAP_OUTSIDE:
                    pixels[x, y] = ImageHandler.__get_color__(COLOR_MAP_OUTSIDE, colors)
                elif pixel_type == ImageHandlerIjai.MAP_WALL:
                    pixels[x, y] = ImageHandler.__get_color__(COLOR_MAP_WALL_V2, colors)
                elif pixel_type == ImageHandlerIjai.MAP_SCAN:
                    pixels[x, y] = ImageHandler.__get_color__(COLOR_SCAN, colors)
                elif pixel_type == ImageHandlerIjai.MAP_NEW_DISCOVERED_AREA:
                    pixels[x, y] = ImageHandler.__get_color__(COLOR_NEW_DISCOVERED_AREA, colors)
                elif ImageHandlerIjai.MAP_ROOM_MIN <= pixel_type <= ImageHandlerIjai.MAP_SELECTED_ROOM_MAX:
                    room_x = img_x + trim_left
                    room_y = img_y + trim_bottom
                    room_number = pixel_type
                    if pixel_type >= ImageHandlerIjai.MAP_SELECTED_ROOM_MIN:
                        room_number = pixel_type - ImageHandlerIjai.MAP_SELECTED_ROOM_MIN + ImageHandlerIjai.MAP_ROOM_MIN
                        cleaned_areas.add(room_number)
                        if draw_cleaned_area:
                            cleaned_areas_pixels[x, y] = ImageHandler.__get_color__(COLOR_CLEANED_AREA, colors)
                    rooms[room_number] = (room_x, room_y, room_x, room_y) \
                    if room_number not in rooms \
                    else (min(rooms[room_number][0], room_x),
                        min(rooms[room_number][1], room_y),
                        max(rooms[room_number][2], room_x),
                        max(rooms[room_number][3], room_y))
                    default = ImageHandler.ROOM_COLORS[room_number % len(ImageHandler.ROOM_COLORS)]
                    pixels[x, y] = ImageHandler.__get_color__(f"{COLOR_ROOM_PREFIX}{room_number}", colors, default)
                else:
                    pixels[x, y] = ImageHandler.__get_color__(COLOR_UNKNOWN, colors)
                    unknown_pixels.add(pixel_type)
                    _LOGGER.debug(f"unknown pixel [{x},{y}] = {pixel_type}")
        if image_config["scale"] != 1 and trimmed_width != 0 and trimmed_height != 0:
            image = image.resize((int(trimmed_width * scale), int(trimmed_height * scale)), resample=Resampling.NEAREST)
            if draw_cleaned_area:
                cleaned_areas_layer = cleaned_areas_layer.resize(
                    (int(trimmed_width * scale), int(trimmed_height * scale)), resample=Image.NEAREST)
        if len(unknown_pixels) > 0:
            _LOGGER.warning('unknown pixel_types: %s', unknown_pixels)
        return image, rooms, cleaned_areas, cleaned_areas_layer

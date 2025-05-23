"""Constants for Xiaomi Cloud Map Extractor."""

DOMAIN = "xiaomi_cloud_map_extractor"
PLATFORMS = ["camera"]

CONF_ATTRIBUTES = "attributes"
CONF_AUTO_UPDATE = "auto_update"
CONF_AVAILABLE_API_DREAME = "dreame"
CONF_AVAILABLE_API_ROIDMI = "roidmi"
CONF_AVAILABLE_API_VIOMI = "viomi"
CONF_AVAILABLE_API_XIAOMI = "xiaomi"
CONF_AVAILABLE_API_IJAI = "ijai"
CONF_AVAILABLE_COUNTRIES = ["cn", "de", "us", "ru", "tw", "sg", "in", "i2"]
CONF_BOTTOM = "bottom"
CONF_COLOR = "color"
CONF_COLORS = "colors"
CONF_COUNTRY = "country"
CONF_DRAW = "draw"
CONF_FORCE_API = "force_api"
CONF_FONT = "font"
CONF_FONT_SIZE = "font_size"
CONF_LEFT = "left"
CONF_MAP_TRANSFORM = "map_transformation"
CONF_RIGHT = "right"
CONF_ROOM_COLORS = "room_colors"
CONF_ROTATE = "rotate"
CONF_SCALE = "scale"
CONF_SIZES = "sizes"
CONF_SIZE_CHARGER_RADIUS = "charger_radius"
CONF_SIZE_IGNORED_OBSTACLE_RADIUS = "ignored_obstacle_radius"
CONF_SIZE_IGNORED_OBSTACLE_WITH_PHOTO_RADIUS = "ignored_obstacle_with_photo_radius"
CONF_SIZE_MOP_PATH_WIDTH = "mop_path_width"
CONF_SIZE_OBSTACLE_RADIUS = "obstacle_radius"
CONF_SIZE_OBSTACLE_WITH_PHOTO_RADIUS = "obstacle_with_photo_radius"
CONF_SIZE_VACUUM_RADIUS = "vacuum_radius"
CONF_SIZE_PATH_WIDTH = "path_width"
CONF_STORE_MAP_RAW = "store_map_raw"
CONF_STORE_MAP_IMAGE = "store_map_image"
CONF_STORE_MAP_PATH = "store_map_path"
CONF_TEXT = "text"
CONF_TEXTS = "texts"
CONF_TOP = "top"
CONF_TRIM = "trim"
CONF_X = "x"
CONF_Y = "y"
CONF_VACUUM_ENTITY_ID = "vacuum_entity_id" # New constant

CONF_AVAILABLE_APIS = [CONF_AVAILABLE_API_XIAOMI, CONF_AVAILABLE_API_VIOMI, CONF_AVAILABLE_API_ROIDMI,
                       CONF_AVAILABLE_API_DREAME, CONF_AVAILABLE_API_IJAI]

CONF_AVAILABLE_SIZES = [CONF_SIZE_VACUUM_RADIUS, CONF_SIZE_PATH_WIDTH, CONF_SIZE_IGNORED_OBSTACLE_RADIUS,
                        CONF_SIZE_IGNORED_OBSTACLE_WITH_PHOTO_RADIUS, CONF_SIZE_MOP_PATH_WIDTH,
                        CONF_SIZE_OBSTACLE_RADIUS, CONF_SIZE_OBSTACLE_WITH_PHOTO_RADIUS,
                        CONF_SIZE_CHARGER_RADIUS]

MINIMAL_IMAGE_WIDTH = 20
MINIMAL_IMAGE_HEIGHT = 20
CONTENT_TYPE = "image/png"
DEFAULT_NAME = "Xiaomi Cloud Map Extractor"

ATTRIBUTE_CALIBRATION = "calibration_points"
ATTRIBUTE_CARPET_MAP = "carpet_map"
ATTRIBUTE_CHARGER = "charger"
ATTRIBUTE_CLEANED_ROOMS = "cleaned_rooms"
ATTRIBUTE_COUNTRY = "country"
ATTRIBUTE_GOTO = "goto"
ATTRIBUTE_GOTO_PATH = "goto_path"
ATTRIBUTE_GOTO_PREDICTED_PATH = "goto_predicted_path"
ATTRIBUTE_IGNORED_OBSTACLES = "ignored_obstacles"
ATTRIBUTE_IGNORED_OBSTACLES_WITH_PHOTO = "ignored_obstacles_with_photo"
ATTRIBUTE_IMAGE = "image"
ATTRIBUTE_IS_EMPTY = "is_empty"
ATTRIBUTE_MAP_NAME = "map_name"
ATTRIBUTE_MOP_PATH = "mop_path"
ATTRIBUTE_MAP_SAVED = "map_saved"
ATTRIBUTE_NO_CARPET_AREAS = "no_carpet_areas"
ATTRIBUTE_NO_GO_AREAS = "no_go_areas"
ATTRIBUTE_NO_MOPPING_AREAS = "no_mopping_areas"
ATTRIBUTE_OBSTACLES = "obstacles"
ATTRIBUTE_OBSTACLES_WITH_PHOTO = "obstacles_with_photo"
ATTRIBUTE_PATH = "path"
ATTRIBUTE_ROOMS = "rooms"
ATTRIBUTE_ROOM_NUMBERS = "room_numbers"
ATTRIBUTE_VACUUM_POSITION = "vacuum_position"
ATTRIBUTE_VACUUM_ROOM = "vacuum_room"
ATTRIBUTE_VACUUM_ROOM_NAME = "vacuum_room_name"
ATTRIBUTE_WALLS = "walls"
ATTRIBUTE_ZONES = "zones"

CONF_AVAILABLE_ATTRIBUTES = [ATTRIBUTE_CALIBRATION, ATTRIBUTE_CARPET_MAP, ATTRIBUTE_NO_CARPET_AREAS,
                             ATTRIBUTE_CHARGER, ATTRIBUTE_CLEANED_ROOMS, ATTRIBUTE_COUNTRY,
                             ATTRIBUTE_GOTO, ATTRIBUTE_GOTO_PATH, ATTRIBUTE_GOTO_PREDICTED_PATH,
                             ATTRIBUTE_IGNORED_OBSTACLES, ATTRIBUTE_IGNORED_OBSTACLES_WITH_PHOTO, ATTRIBUTE_IMAGE,
                             ATTRIBUTE_IS_EMPTY, ATTRIBUTE_MAP_NAME, ATTRIBUTE_MOP_PATH, ATTRIBUTE_NO_GO_AREAS,
                             ATTRIBUTE_NO_MOPPING_AREAS, ATTRIBUTE_OBSTACLES, ATTRIBUTE_OBSTACLES_WITH_PHOTO,
                             ATTRIBUTE_PATH, ATTRIBUTE_ROOMS, ATTRIBUTE_ROOM_NUMBERS, ATTRIBUTE_VACUUM_POSITION,
                             ATTRIBUTE_VACUUM_ROOM, ATTRIBUTE_VACUUM_ROOM_NAME, ATTRIBUTE_WALLS, ATTRIBUTE_ZONES]

ATTR_A = "a"
ATTR_ANGLE = "angle"
ATTR_CONFIDENCE_LEVEL = "confidence_level"
ATTR_DESCRIPTION = "description"
ATTR_HEIGHT = "height"
ATTR_MODEL = "model"
ATTR_NAME = "name"
ATTR_OFFSET_X = "offset_x"
ATTR_OFFSET_Y = "offset_y"
ATTR_PATH = "path"
ATTR_PHOTO_NAME = "photo_name"
ATTR_POINT_LENGTH = "point_length"
ATTR_POINT_SIZE = "point_size"
ATTR_ROTATION = "rotation"
ATTR_SCALE = "scale"
ATTR_SIZE = "size"
ATTR_TWO_FACTOR_AUTH = "url_2fa"
ATTR_TYPE = "type"
ATTR_USED_API = "used_api"
ATTR_WIDTH = "width"
ATTR_X = "x"
ATTR_X0 = "x0"
ATTR_X1 = "x1"
ATTR_X2 = "x2"
ATTR_X3 = "x3"
ATTR_Y = "y"
ATTR_Y0 = "y0"
ATTR_Y1 = "y1"
ATTR_Y2 = "y2"
ATTR_Y3 = "y3"

DRAWABLE_ALL = "all"
DRAWABLE_CHARGER = "charger"
DRAWABLE_CLEANED_AREA = "cleaned_area"
DRAWABLE_GOTO_PATH = "goto_path"
DRAWABLE_IGNORED_OBSTACLES = "ignored_obstacles"
DRAWABLE_IGNORED_OBSTACLES_WITH_PHOTO = "ignored_obstacles_with_photo"
DRAWABLE_MOP_PATH = "mop_path"
DRAWABLE_NO_CARPET_AREAS = "no_carpet_zones"
DRAWABLE_NO_GO_AREAS = "no_go_zones"
DRAWABLE_NO_MOPPING_AREAS = "no_mopping_zones"
DRAWABLE_OBSTACLES = "obstacles"
DRAWABLE_OBSTACLES_WITH_PHOTO = "obstacles_with_photo"
DRAWABLE_PATH = "path"
DRAWABLE_PREDICTED_PATH = "predicted_path"
DRAWABLE_ROOM_NAMES = "room_names"
DRAWABLE_VACUUM_POSITION = "vacuum_position"
DRAWABLE_VIRTUAL_WALLS = "virtual_walls"
DRAWABLE_ZONES = "zones"

CONF_AVAILABLE_DRAWABLES = [DRAWABLE_ALL, DRAWABLE_CLEANED_AREA, DRAWABLE_CHARGER, DRAWABLE_GOTO_PATH,
                            DRAWABLE_IGNORED_OBSTACLES, DRAWABLE_IGNORED_OBSTACLES_WITH_PHOTO, DRAWABLE_MOP_PATH,
                            DRAWABLE_NO_CARPET_AREAS, DRAWABLE_NO_GO_AREAS, DRAWABLE_NO_MOPPING_AREAS, DRAWABLE_OBSTACLES,
                            DRAWABLE_OBSTACLES_WITH_PHOTO, DRAWABLE_PATH, DRAWABLE_PREDICTED_PATH, DRAWABLE_ROOM_NAMES,
                            DRAWABLE_VACUUM_POSITION, DRAWABLE_VIRTUAL_WALLS, DRAWABLE_ZONES]

COLOR_ROOM_PREFIX = "color_room_"

COLOR_CARPETS = "color_carpets"
COLOR_CHARGER = "color_charger"
COLOR_CHARGER_OUTLINE = "color_charger_outline"
COLOR_CLEANED_AREA = "color_cleaned_area"
COLOR_GOTO_PATH = "color_goto_path"
COLOR_GREY_WALL = "color_grey_wall"
COLOR_IGNORED_OBSTACLE = "color_ignored_obstacle"
COLOR_IGNORED_OBSTACLE_WITH_PHOTO = "color_ignored_obstacle_with_photo"
COLOR_MAP_INSIDE = "color_map_inside"
COLOR_MAP_OUTSIDE = "color_map_outside"
COLOR_MAP_WALL = "color_map_wall"
COLOR_MAP_WALL_V2 = "color_map_wall_v2"
COLOR_MOP_PATH = "color_mop_path"
COLOR_NEW_DISCOVERED_AREA = "color_new_discovered_area"
COLOR_NO_CARPET_ZONES = "color_no_carpet_zones"
COLOR_NO_CARPET_ZONES_OUTLINE = "color_no_carpet_zones_outline"
COLOR_NO_GO_ZONES = "color_no_go_zones"
COLOR_NO_GO_ZONES_OUTLINE = "color_no_go_zones_outline"
COLOR_NO_MOPPING_ZONES = "color_no_mop_zones"
COLOR_NO_MOPPING_ZONES_OUTLINE = "color_no_mop_zones_outline"
COLOR_OBSTACLE = "color_obstacle"
COLOR_OBSTACLE_WITH_PHOTO = "color_obstacle_with_photo"
COLOR_PATH = "color_path"
COLOR_PREDICTED_PATH = "color_predicted_path"
COLOR_ROBO = "color_robo"
COLOR_ROBO_OUTLINE = "color_robo_outline"
COLOR_ROOM_NAMES = "color_room_names"
COLOR_SCAN = "color_scan"
COLOR_UNKNOWN = "color_unknown"
COLOR_VIRTUAL_WALLS = "color_virtual_walls"
COLOR_ZONES = "color_zones"
COLOR_ZONES_OUTLINE = "color_zones_outline"

CONF_AVAILABLE_COLORS = [COLOR_CARPETS, COLOR_CHARGER, COLOR_CHARGER_OUTLINE, COLOR_CLEANED_AREA, COLOR_GOTO_PATH,
                         COLOR_GREY_WALL, COLOR_IGNORED_OBSTACLE, COLOR_IGNORED_OBSTACLE_WITH_PHOTO, COLOR_MAP_INSIDE, COLOR_MAP_OUTSIDE,
                         COLOR_MAP_WALL, COLOR_MAP_WALL_V2, COLOR_MOP_PATH, COLOR_NEW_DISCOVERED_AREA, COLOR_NO_CARPET_ZONES,
                         COLOR_NO_CARPET_ZONES_OUTLINE, COLOR_NO_GO_ZONES, COLOR_NO_GO_ZONES_OUTLINE,
                         COLOR_NO_MOPPING_ZONES, COLOR_NO_MOPPING_ZONES_OUTLINE, COLOR_OBSTACLE, COLOR_OBSTACLE_WITH_PHOTO,
                         COLOR_PATH, COLOR_PREDICTED_PATH, COLOR_ROBO, COLOR_ROBO_OUTLINE, COLOR_ROOM_NAMES,
                         COLOR_SCAN, COLOR_UNKNOWN, COLOR_VIRTUAL_WALLS, COLOR_ZONES, COLOR_ZONES_OUTLINE]

COLOR_ROOM_1 = "color_room_1"
COLOR_ROOM_2 = "color_room_2"
COLOR_ROOM_3 = "color_room_3"
COLOR_ROOM_4 = "color_room_4"
COLOR_ROOM_5 = "color_room_5"
COLOR_ROOM_6 = "color_room_6"
COLOR_ROOM_7 = "color_room_7"
COLOR_ROOM_8 = "color_room_8"
COLOR_ROOM_9 = "color_room_9"
COLOR_ROOM_10 = "color_room_10"
COLOR_ROOM_11 = "color_room_11"
COLOR_ROOM_12 = "color_room_12"
COLOR_ROOM_13 = "color_room_13"
COLOR_ROOM_14 = "color_room_14"
COLOR_ROOM_15 = "color_room_15"
COLOR_ROOM_16 = "color_room_16"

CONF_DEFAULT_ROOM_COLORS = [COLOR_ROOM_1, COLOR_ROOM_2, COLOR_ROOM_3, COLOR_ROOM_4, COLOR_ROOM_5, COLOR_ROOM_6,
                            COLOR_ROOM_7, COLOR_ROOM_8, COLOR_ROOM_9, COLOR_ROOM_10, COLOR_ROOM_11, COLOR_ROOM_12,
                            COLOR_ROOM_13, COLOR_ROOM_14, COLOR_ROOM_15, COLOR_ROOM_16]

AVAILABLE_APIS = {
    CONF_AVAILABLE_API_DREAME: ["dreame.vacuum."],
    CONF_AVAILABLE_API_ROIDMI: ["roidmi.vacuum.", "zhimi.vacuum.", "chuangmi.vacuum."],
    CONF_AVAILABLE_API_VIOMI: ["viomi.vacuum."],
    CONF_AVAILABLE_API_XIAOMI: ["roborock.vacuum", "rockrobo.vacuum"],
    CONF_AVAILABLE_API_IJAI: ["ijai.vacuum."]
}

API_EXCEPTIONS = {
    "viomi.vacuum.v18": CONF_AVAILABLE_API_ROIDMI,
    "viomi.vacuum.v23": CONF_AVAILABLE_API_ROIDMI,
    "viomi.vacuum.v38": CONF_AVAILABLE_API_ROIDMI,
}

MM = 50

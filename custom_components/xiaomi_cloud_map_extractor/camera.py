import io
import logging
import time
import os # Still useful for _store_image_if_enabled if user wants that
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from custom_components.xiaomi_cloud_map_extractor.common.backoff import Backoff
from custom_components.xiaomi_cloud_map_extractor.common.map_data import MapData
from custom_components.xiaomi_cloud_map_extractor.common.vacuum import XiaomiCloudVacuum
from custom_components.xiaomi_cloud_map_extractor.types import Colors, Drawables, ImageConfig, Sizes, Texts

try:
    from miio import RoborockVacuum, DeviceException
except ImportError:
    from miio import Vacuum as RoborockVacuum, DeviceException

import PIL.Image as Image
import voluptuous as vol
from homeassistant.components import vacuum
from homeassistant.components.camera import Camera, CameraEntityFeature, ENTITY_ID_FORMAT, PLATFORM_SCHEMA

from homeassistant.const import CONF_ENTITY_ID, CONF_HOST, CONF_NAME, CONF_PASSWORD, CONF_TOKEN, CONF_USERNAME
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity import generate_entity_id
from homeassistant.helpers.reload import async_setup_reload_service

from custom_components.xiaomi_cloud_map_extractor.common.map_data_parser import MapDataParser
from custom_components.xiaomi_cloud_map_extractor.common.xiaomi_cloud_connector import XiaomiCloudConnector
from custom_components.xiaomi_cloud_map_extractor.const import *
from custom_components.xiaomi_cloud_map_extractor.dreame.vacuum import DreameVacuum
from custom_components.xiaomi_cloud_map_extractor.roidmi.vacuum import RoidmiVacuum
from custom_components.xiaomi_cloud_map_extractor.unsupported.vacuum import UnsupportedVacuum
from custom_components.xiaomi_cloud_map_extractor.viomi.vacuum import ViomiVacuum
from custom_components.xiaomi_cloud_map_extractor.xiaomi.vacuum import XiaomiVacuum
from custom_components.xiaomi_cloud_map_extractor.ijai.vacuum import IjaiVacuum


_LOGGER = logging.getLogger(__name__)

SCAN_INTERVAL = timedelta(seconds=5)

DEFAULT_TRIMS = {
    CONF_LEFT: 0,
    CONF_RIGHT: 0,
    CONF_TOP: 0,
    CONF_BOTTOM: 0
}

'''
DEFAULT_TRIMS = {
    CONF_LEFT: 30,
    CONF_RIGHT: 40,
    CONF_TOP: 28,
    CONF_BOTTOM: 40
}
'''

DEFAULT_SIZES = {
    CONF_SIZE_VACUUM_RADIUS: 6,
    CONF_SIZE_PATH_WIDTH: 1,
    CONF_SIZE_MOP_PATH_WIDTH: 6,
    CONF_SIZE_IGNORED_OBSTACLE_RADIUS: 3,
    CONF_SIZE_IGNORED_OBSTACLE_WITH_PHOTO_RADIUS: 3,
    CONF_SIZE_OBSTACLE_RADIUS: 3,
    CONF_SIZE_OBSTACLE_WITH_PHOTO_RADIUS: 3,
    CONF_SIZE_CHARGER_RADIUS: 6
}

COLOR_SCHEMA = vol.Or(
    vol.All(vol.Length(min=3, max=3), vol.ExactSequence((cv.byte, cv.byte, cv.byte)), vol.Coerce(tuple)),
    vol.All(vol.Length(min=4, max=4), vol.ExactSequence((cv.byte, cv.byte, cv.byte, cv.byte)), vol.Coerce(tuple))
)

PERCENT_SCHEMA = vol.All(vol.Coerce(float), vol.Range(min=0, max=100))

POSITIVE_FLOAT_SCHEMA = vol.All(vol.Coerce(float), vol.Range(min=0))

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_HOST): cv.string,
        vol.Required(CONF_TOKEN): vol.All(str, vol.Length(min=32, max=32)),
        vol.Required(CONF_USERNAME): cv.string,
        vol.Required(CONF_PASSWORD): cv.string,
        vol.Optional(CONF_COUNTRY, default=None): vol.Or(vol.In(CONF_AVAILABLE_COUNTRIES), vol.Equal(None)),
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_AUTO_UPDATE, default=True): cv.boolean,
        vol.Optional(CONF_COLORS, default={}): vol.Schema({
            vol.In(CONF_AVAILABLE_COLORS): COLOR_SCHEMA
        }),
        vol.Optional(CONF_ROOM_COLORS, default={}): vol.Schema({
            cv.positive_int: COLOR_SCHEMA
        }),
        vol.Optional(CONF_DRAW, default=[]): vol.All(cv.ensure_list, [vol.In(CONF_AVAILABLE_DRAWABLES)]),
        vol.Optional(CONF_MAP_TRANSFORM, default={CONF_SCALE: 1, CONF_ROTATE: 0, CONF_TRIM: DEFAULT_TRIMS}):
            vol.Schema({
                vol.Optional(CONF_SCALE, default=1): POSITIVE_FLOAT_SCHEMA,
                vol.Optional(CONF_ROTATE, default=0): vol.In([0, 90, 180, 270]),
                vol.Optional(CONF_TRIM, default=DEFAULT_TRIMS): vol.Schema({
                    vol.Optional(CONF_LEFT, default=0): PERCENT_SCHEMA,
                    vol.Optional(CONF_RIGHT, default=0): PERCENT_SCHEMA,
                    vol.Optional(CONF_TOP, default=0): PERCENT_SCHEMA,
                    vol.Optional(CONF_BOTTOM, default=0): PERCENT_SCHEMA
                }),
            }),
        vol.Optional(CONF_ATTRIBUTES, default=[]): vol.All(cv.ensure_list, [vol.In(CONF_AVAILABLE_ATTRIBUTES)]),
        vol.Optional(CONF_TEXTS, default=[]):
            vol.All(cv.ensure_list, [vol.Schema({
                vol.Required(CONF_TEXT): cv.string,
                vol.Required(CONF_X): vol.Coerce(float),
                vol.Required(CONF_Y): vol.Coerce(float),
                vol.Optional(CONF_COLOR, default=(0, 0, 0)): COLOR_SCHEMA,
                vol.Optional(CONF_FONT, default=None): vol.Or(cv.string, vol.Equal(None)),
                vol.Optional(CONF_FONT_SIZE, default=0): cv.positive_int
            })]),
        vol.Optional(CONF_SIZES, default=DEFAULT_SIZES): vol.Schema({
            vol.Optional(CONF_SIZE_VACUUM_RADIUS,
                         default=DEFAULT_SIZES[CONF_SIZE_VACUUM_RADIUS]): POSITIVE_FLOAT_SCHEMA,
            vol.Optional(CONF_SIZE_PATH_WIDTH,
                         default=DEFAULT_SIZES[CONF_SIZE_PATH_WIDTH]): POSITIVE_FLOAT_SCHEMA,
            vol.Optional(CONF_SIZE_MOP_PATH_WIDTH, 
                         default=DEFAULT_SIZES[CONF_SIZE_MOP_PATH_WIDTH]): POSITIVE_FLOAT_SCHEMA,
            vol.Optional(CONF_SIZE_IGNORED_OBSTACLE_RADIUS,
                         default=DEFAULT_SIZES[CONF_SIZE_IGNORED_OBSTACLE_RADIUS]): POSITIVE_FLOAT_SCHEMA,
            vol.Optional(CONF_SIZE_IGNORED_OBSTACLE_WITH_PHOTO_RADIUS,
                         default=DEFAULT_SIZES[CONF_SIZE_IGNORED_OBSTACLE_WITH_PHOTO_RADIUS]): POSITIVE_FLOAT_SCHEMA,
            vol.Optional(CONF_SIZE_OBSTACLE_RADIUS,
                         default=DEFAULT_SIZES[CONF_SIZE_OBSTACLE_RADIUS]): POSITIVE_FLOAT_SCHEMA,
            vol.Optional(CONF_SIZE_OBSTACLE_WITH_PHOTO_RADIUS,
                         default=DEFAULT_SIZES[CONF_SIZE_OBSTACLE_WITH_PHOTO_RADIUS]): POSITIVE_FLOAT_SCHEMA,
            vol.Optional(CONF_SIZE_CHARGER_RADIUS,
                         default=DEFAULT_SIZES[CONF_SIZE_CHARGER_RADIUS]): POSITIVE_FLOAT_SCHEMA
        }),
        # store_map_raw and store_map_path are kept for users who want to save maps for other purposes
        vol.Optional(CONF_STORE_MAP_RAW, default=False): cv.boolean,
        vol.Optional(CONF_STORE_MAP_IMAGE, default=False): cv.boolean,
        vol.Optional(CONF_STORE_MAP_PATH, default="/tmp"): cv.string,
        vol.Optional(CONF_FORCE_API, default=None): vol.Or(vol.In(CONF_AVAILABLE_APIS), vol.Equal(None)),
        vol.Optional(CONF_VACUUM_ENTITY_ID): cv.entity_id, 
    })

ROOMS_CLEANING_SCHEMA = vol.Schema({
    vol.Required(CONF_ENTITY_ID): cv.entity_domain(vacuum.DOMAIN),
    vol.Required("rooms_id"): cv.string, 
})


async def async_setup_platform(hass, config, async_add_entities, discovery_info=None):
    await async_setup_reload_service(hass, DOMAIN, PLATFORMS)

    async def handle_rooms_cleaning(call):
        rooms_id_str = call.data.get("rooms_id") 
        entity_id = call.data.get("entity_id")

        if not entity_id:
            _LOGGER.error("No entity_id was provided to start the cleaning.")
            return
        
        siid = 7 
        aiid = 3 
        
        params = []
        if rooms_id_str:
            params.append(rooms_id_str) 
            params.append(0) 
            params.append(1) 
            _LOGGER.debug(f"Calling MIoT action for {entity_id} with SIID={siid}, AIID={aiid}, PARAMS={params}")

        try:
            await hass.services.async_call(
                "xiaomi_miot", 
                "call_action",
                {
                    "entity_id": entity_id,
                    "siid": siid,
                    "aiid": aiid,
                    "params": params,
                },
                blocking=True 
            )
            _LOGGER.info(f"MIoT action call for room cleaning sent to {entity_id}.")
        except Exception as e:
            _LOGGER.error(f"Error calling MIoT action for {entity_id}: {e}")

    hass.services.async_register(DOMAIN, 'rooms_cleaning', handle_rooms_cleaning, schema=ROOMS_CLEANING_SCHEMA)

    host = config[CONF_HOST]
    token = config[CONF_TOKEN]
    username = config[CONF_USERNAME]
    password = config[CONF_PASSWORD]
    country = config[CONF_COUNTRY]
    name = config[CONF_NAME]
    should_poll = config[CONF_AUTO_UPDATE]
    image_config = config[CONF_MAP_TRANSFORM]
    colors = config[CONF_COLORS].copy() 
    room_colors = config[CONF_ROOM_COLORS]
    for room, color in room_colors.items():
        colors[f"{COLOR_ROOM_PREFIX}{room}"] = color
    drawables = list(config[CONF_DRAW]) 
    sizes = config[CONF_SIZES]
    texts = config[CONF_TEXTS]
    if DRAWABLE_ALL in drawables:
        drawables = [d for d in CONF_AVAILABLE_DRAWABLES if d != DRAWABLE_ALL] 

    attributes = config[CONF_ATTRIBUTES]
    store_map_raw = config[CONF_STORE_MAP_RAW]
    store_map_image = config[CONF_STORE_MAP_IMAGE]
    store_map_path = config[CONF_STORE_MAP_PATH]
    force_api = config.get(CONF_FORCE_API) 
    vacuum_entity_id = config.get(CONF_VACUUM_ENTITY_ID) 

    entity_id = generate_entity_id(ENTITY_ID_FORMAT, name, hass=hass)
    async_add_entities([VacuumCamera(entity_id, host, token, username, password, country, name, should_poll,
                                     image_config, colors, drawables, sizes, texts, attributes, store_map_raw,
                                     store_map_image, store_map_path, force_api, vacuum_entity_id)])


class VacuumCamera(Camera):
    _map_name: Optional[str] = None

    def __init__(self, entity_id: str, host: str, token: str, username: str, password: str, country: str, name: str,
                 should_poll: bool, image_config: ImageConfig, colors: Colors, drawables: Drawables, sizes: Sizes,
                 texts: Texts, attributes: List[str], store_map_raw: bool, store_map_image: bool,
                 store_map_path: str, force_api: Optional[str], vacuum_entity_id: Optional[str] = None):
        super().__init__()
        self.entity_id = entity_id
        self.content_type = CONTENT_TYPE
        self._vacuum_device_obj = RoborockVacuum(host, token) 
        self._connector = XiaomiCloudConnector(username, password)
        self._status = CameraStatus.INITIALIZING
        self._device: Optional[XiaomiCloudVacuum] = None
        self._host = host
        self._token = token 
        self._name = name
        self._should_poll = should_poll
        self._image_config = image_config
        self._colors = colors
        self._drawables = drawables
        self._sizes = sizes
        self._texts = texts
        self._attributes = attributes
        self._store_map_raw = store_map_raw # Kept for users who want to save maps
        self._store_map_image = store_map_image
        self._store_map_path = store_map_path
        self._forced_api = force_api
        self._vacuum_entity_id = vacuum_entity_id
        self._used_api = None
        self._map_saved: Optional[bool] = None 
        self._image: Optional[bytes] = None
        self._map_data: Optional[MapData] = None
        self._logged_in = False
        self._logged_in_previously = True 
        self._country = country

    async def async_added_to_hass(self) -> None:
        """Handle when entity is added."""
        await super().async_added_to_hass() 
        if self._should_poll:
            self.async_schedule_update_ha_state(True)


    @property
    def frame_interval(self) -> float:
        return 1.0 

    def camera_image(self, width: Optional[int] = None, height: Optional[int] = None) -> Optional[bytes]:
        return self._image

    @property
    def name(self) -> str:
        return self._name

    def turn_on(self):
        self._should_poll = True
        self.async_schedule_update_ha_state(True) 


    def turn_off(self):
        self._should_poll = False

    @property
    def supported_features(self) -> int: 
        return CameraEntityFeature.ON_OFF

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        attributes = {}
        if self._map_data is not None:
            attributes.update(self.extract_attributes(self._map_data, self._attributes, self._country))
        
        if self._store_map_raw: 
             attributes[ATTRIBUTE_MAP_SAVED] = self._map_saved

        if self._device is not None:
            attributes[ATTR_MODEL] = self._device.model
            attributes[ATTR_USED_API] = self._used_api
        if self._connector.two_factor_auth_url is not None:
            attributes[ATTR_TWO_FACTOR_AUTH] = self._connector.two_factor_auth_url
        
        attributes["camera_status"] = str(self._status)
        return attributes

    @property
    def should_poll(self) -> bool:
        return self._should_poll

    @staticmethod
    def extract_attributes(map_data: MapData, attributes_to_return: List[str], country) -> Dict[str, Any]:
        attributes = {}
        rooms_attr_list = []
        if map_data.rooms:
            for room_id, room_obj in sorted(map_data.rooms.items()): 
                room_info = {"id": room_id}
                if room_obj.name:
                    room_info["name"] = room_obj.name
                else:
                    room_info["name"] = f"Room {room_id}" 
                rooms_attr_list.append(room_info)

        attribute_sources = {
            ATTRIBUTE_CALIBRATION: lambda: map_data.calibration(),
            ATTRIBUTE_CARPET_MAP: lambda: list(map_data.carpet_map) if map_data.carpet_map else [], 
            ATTRIBUTE_CHARGER: lambda: map_data.charger.as_dict() if map_data.charger else None,
            ATTRIBUTE_CLEANED_ROOMS: lambda: list(map_data.cleaned_rooms) if map_data.cleaned_rooms else [], 
            ATTRIBUTE_COUNTRY: lambda: country,
            ATTRIBUTE_GOTO: lambda: [p.as_dict() for p in map_data.goto] if map_data.goto else None,
            ATTRIBUTE_GOTO_PATH: lambda: map_data.goto_path.as_dict() if map_data.goto_path else None,
            ATTRIBUTE_GOTO_PREDICTED_PATH: lambda: map_data.predicted_path.as_dict() if map_data.predicted_path else None,
            ATTRIBUTE_IGNORED_OBSTACLES: lambda: [obs.as_dict() for obs in map_data.ignored_obstacles] if map_data.ignored_obstacles else [],
            ATTRIBUTE_IGNORED_OBSTACLES_WITH_PHOTO: lambda: [obs.as_dict() for obs in map_data.ignored_obstacles_with_photo] if map_data.ignored_obstacles_with_photo else [],
            ATTRIBUTE_IMAGE: lambda: map_data.image.as_dict() if map_data.image else None,
            ATTRIBUTE_IS_EMPTY: lambda: map_data.image.is_empty if map_data.image else True,
            ATTRIBUTE_MAP_NAME: lambda: map_data.map_name,
            ATTRIBUTE_MOP_PATH: lambda: map_data.mop_path.as_dict() if map_data.mop_path else None,
            ATTRIBUTE_NO_CARPET_AREAS: lambda: [area.as_dict() for area in map_data.no_carpet_areas] if map_data.no_carpet_areas else [],
            ATTRIBUTE_NO_GO_AREAS: lambda: [area.as_dict() for area in map_data.no_go_areas] if map_data.no_go_areas else [],
            ATTRIBUTE_NO_MOPPING_AREAS: lambda: [area.as_dict() for area in map_data.no_mopping_areas] if map_data.no_mopping_areas else [],
            ATTRIBUTE_OBSTACLES: lambda: [obs.as_dict() for obs in map_data.obstacles] if map_data.obstacles else [],
            ATTRIBUTE_OBSTACLES_WITH_PHOTO: lambda: [obs.as_dict() for obs in map_data.obstacles_with_photo] if map_data.obstacles_with_photo else [],
            ATTRIBUTE_PATH: lambda: map_data.path.as_dict() if map_data.path else None,
            ATTRIBUTE_ROOM_NUMBERS: lambda: rooms_attr_list, 
            ATTRIBUTE_ROOMS: lambda: {k: v.as_dict() for k, v in map_data.rooms.items()} if map_data.rooms else {},
            ATTRIBUTE_VACUUM_POSITION: lambda: map_data.vacuum_position.as_dict() if map_data.vacuum_position else None,
            ATTRIBUTE_VACUUM_ROOM: lambda: map_data.vacuum_room,
            ATTRIBUTE_VACUUM_ROOM_NAME: lambda: map_data.vacuum_room_name,
            ATTRIBUTE_WALLS: lambda: [wall.as_dict() for wall in map_data.walls] if map_data.walls else [],
            ATTRIBUTE_ZONES: lambda: [zone.as_dict() for zone in map_data.zones] if map_data.zones else []
        }

        for attr_name in attributes_to_return:
            if attr_name in attribute_sources:
                try:
                    attributes[attr_name] = attribute_sources[attr_name]()
                except Exception as e:
                    _LOGGER.warning(f"Error extracting attribute {attr_name}: {e}")
                    attributes[attr_name] = None 
        return attributes

    def _handle_map_data_cloud_fetch(self, map_name: str):
        """Handles fetching map data from Xiaomi cloud and processing it."""
        _LOGGER.debug("Retrieving map from Xiaomi cloud for map_name: %s", map_name)
        store_map_path_if_enabled = self._store_map_path if self._store_map_raw else None
        
        map_data, map_stored_successfully = self._device.get_map(
            map_name, self._colors, self._drawables, self._texts,
            self._sizes, self._image_config, store_map_path_if_enabled
        )
        
        self._map_saved = map_stored_successfully 

        if map_data is not None:
            try:
                _LOGGER.debug("Map data retrieved from cloud.")
                if map_data.image is None or map_data.image.is_empty: 
                    _LOGGER.debug("Cloud map is empty or image data is missing.")
                    self._status = CameraStatus.EMPTY_MAP
                    if self._map_data is None or self._map_data.image is None or self._map_data.image.is_empty:
                        self._set_map_data(map_data)
                else:
                    _LOGGER.debug("Cloud map is ok.")
                    self._set_map_data(map_data) 
                    self._status = CameraStatus.OK
            except Exception as e:
                _LOGGER.warning(f"Unable to parse map data from cloud: {e}", exc_info=True)
                self._status = CameraStatus.UNABLE_TO_PARSE_MAP
                if self._map_data is None or self._map_data.image is None or self._map_data.image.is_empty:
                     self._set_map_data(MapDataParser.create_empty(self._colors, str(self._status)))
        else: 
            self._logged_in = False 
            _LOGGER.warning("Unable to retrieve map data from cloud (map_data is None).")
            self._status = CameraStatus.UNABLE_TO_RETRIEVE_MAP
            if self._map_data is None or self._map_data.image is None or self._map_data.image.is_empty:
                self._set_map_data(MapDataParser.create_empty(self._colors, str(self._status)))

    def update(self):
        """Fetch new state data for the camera."""
        counter = 10 

        if self._status != CameraStatus.TWO_FACTOR_AUTH_REQUIRED and not self._logged_in:
            self._handle_login()
        
        if self._device is None and self._logged_in:
            self._handle_device()

        attempt_cloud_fetch = True 
        reused_in_memory_map = False

        # Check if conditions are met to reuse existing in-memory map
        should_try_in_memory_reuse = (
            self._vacuum_entity_id and
            self.hass and
            self._device and # Device must be initialized
            self._map_data and self._map_data.image and not self._map_data.image.is_empty # Must have a valid map in memory
        )

        if should_try_in_memory_reuse:
            try:
                vac_entity = self.hass.states.get(self._vacuum_entity_id)
                # Use 'charging', 'idle', or 'docked' as trigger states to reuse in-memory map
                is_inactive_state = vac_entity and vac_entity.state in ['charging', 'idle', 'docked']
                
                if is_inactive_state:
                    _LOGGER.info(f"Vacuum is in state '{vac_entity.state}'. Reusing existing in-memory map.")
                    # Ensure the image is rendered if it was somehow cleared
                    if not self._image: 
                        self._set_map_data(self._map_data)
                    self._status = CameraStatus.OK # Status is OK as we have a usable map
                    attempt_cloud_fetch = False # Do not fetch from cloud
                    reused_in_memory_map = True
            except Exception as e:
                _LOGGER.warning(f"Error during in-memory map reuse logic: {e}. Proceeding with cloud fetch.", exc_info=True)
                attempt_cloud_fetch = True # Fallback to cloud fetch on any error

        # Force cloud fetch if no map is currently loaded at all, overriding previous decisions
        if self._map_data is None or (self._map_data.image and self._map_data.image.is_empty):
            if not attempt_cloud_fetch: # This means previous logic decided to skip, but we have no map.
                 _LOGGER.info("No valid map currently available (map_data is None or image is empty), overriding to fetch from cloud.")
            attempt_cloud_fetch = True

        # Perform cloud fetch if decided
        if attempt_cloud_fetch:
            _LOGGER.debug("Proceeding with cloud map fetch.")
            
            current_map_name_from_vacuum = self._handle_map_name(counter)
            if current_map_name_from_vacuum != "retry" and current_map_name_from_vacuum is not None:
                self._map_name = current_map_name_from_vacuum
            elif current_map_name_from_vacuum is None: 
                if self._map_name is None: 
                    _LOGGER.warning("Failed to retrieve map name from vacuum and no previous map name known.")
                else: 
                    _LOGGER.warning(f"Failed to retrieve current map name from vacuum. Using last known: {self._map_name}")
            
            # Update status if map name is truly unobtainable and we didn't successfully reuse an in-memory map
            if self._map_name is None and self._device is not None and not reused_in_memory_map:
                self._status = CameraStatus.FAILED_TO_RETRIEVE_MAP_FROM_VACUUM

            if self._logged_in and self._map_name is not None and self._device is not None:
                self._handle_map_data_cloud_fetch(self._map_name) # Actual cloud fetch
            else:
                # This block executes if cloud fetch prerequisites are not met
                # Only update to error map if no map was successfully reused from memory
                if not reused_in_memory_map:
                    _LOGGER.debug("Unable to retrieve new map from cloud (prerequisites not met): LoggedIn=%s, MapName=%s, DeviceExists=%s",
                                  self._logged_in, self._map_name, self._device is not None)
                    if self._map_data is None or (self._map_data.image and self._map_data.image.is_empty):
                        self._set_map_data(MapDataParser.create_empty(self._colors, str(self._status)))
        else:
            # This means attempt_cloud_fetch is False because reused_in_memory_map was true.
            if reused_in_memory_map:
                 _LOGGER.debug("Update cycle completed using existing in-memory map.")
        
        self._logged_in_previously = self._logged_in

    def _handle_login(self):
        """Handles Xiaomi Cloud login."""
        _LOGGER.debug("Logging in to Xiaomi Cloud...")
        self._logged_in = self._connector.login()
        if self._logged_in is None: 
            _LOGGER.warning("Two-factor authentication might be required. Check URL in attributes: %s", 
                            self._connector.two_factor_auth_url)
            self._status = CameraStatus.TWO_FACTOR_AUTH_REQUIRED
        elif self._logged_in:
            _LOGGER.info("Successfully logged in to Xiaomi Cloud.")
            self._status = CameraStatus.LOGGED_IN
        else: 
            if self._logged_in_previously: 
                 _LOGGER.error("Failed to log in to Xiaomi Cloud. Check credentials.")
            else:
                 _LOGGER.debug("Still unable to log in to Xiaomi Cloud.")
            self._status = CameraStatus.FAILED_LOGIN
            
    def _handle_device(self):
        """Retrieves device details from Xiaomi Cloud."""
        if not self._logged_in:
            _LOGGER.warning("Cannot handle device: Not logged in.")
            return
            
        _LOGGER.debug("Retrieving device info. Configured country: %s, vacuum token: %s", self._country, self._token)
        device_details_tuple = self._connector.get_device_details(self._token, self._country)
        
        country, user_id, device_id, model, mac = (None,) * 5 
        if device_details_tuple and len(device_details_tuple) >= 4:
            country = device_details_tuple[0]
            user_id = device_details_tuple[1]
            device_id = device_details_tuple[2]
            model = device_details_tuple[3]
            if len(device_details_tuple) >= 5:
                mac = device_details_tuple[4]

        if model is not None:
            self._country = country 
            _LOGGER.info(f"Retrieved device model: {model}, UserID: {user_id}, DeviceID: {device_id}, Country: {self._country}, MAC: {mac}")
            self._device = self._create_device(user_id, device_id, model, mac)
            _LOGGER.debug(f"Created device wrapper. Used API: {self._used_api}")
        else:
            _LOGGER.error("Failed to retrieve device model using token. Ensure token is correct and vacuum is connected to Xiaomi Cloud.")
            self._status = CameraStatus.FAILED_TO_RETRIEVE_DEVICE

    def _handle_map_name(self, counter: int) -> Optional[str]:
        """Retrieves map name from the vacuum device (local network call)."""
        if self._device is not None and not self._device.should_get_map_from_vacuum():
            return "0" 

        map_name_result: Optional[str] = "retry" 
        i = 1
        backoff = Backoff(min_sleep=0.1, max_sleep=15) 

        while map_name_result == "retry" and i <= counter:
            _LOGGER.debug(f"Requesting map name from vacuum ({self._host})... (Attempt {i}/{counter})")
            try:
                map_info = self._vacuum_device_obj.map() 
                if map_info and isinstance(map_info, list) and len(map_info) > 0:
                    map_name_result = map_info[0]
                elif isinstance(map_info, str): 
                    map_name_result = map_info
                else: 
                    _LOGGER.warning(f"Unexpected format from vacuum.map() call: {map_info}")
                    map_name_result = None 
                    break 

                if map_name_result != "retry" and map_name_result is not None:
                    _LOGGER.debug(f"Received map name: {map_name_result}")
                    return map_name_result
                elif map_name_result is None: 
                    _LOGGER.warning("Vacuum.map() call returned None.")
                    break 
            except OSError as exc:
                _LOGGER.error(f"OSError while fetching map name from vacuum: {exc}")
                return None 
            except DeviceException as exc:
                _LOGGER.warning(f"DeviceException while fetching map name: {exc}. Retrying...")
            
            if map_name_result == "retry": 
                sleep_duration = backoff.backoff()
                _LOGGER.debug(f"Vacuum responded with 'retry' for map name, sleeping for {sleep_duration:.2f}s.")
                time.sleep(sleep_duration)
            i += 1
        
        if map_name_result == "retry": 
            _LOGGER.warning(f"Failed to get a valid map name from vacuum after {counter} retries (kept returning 'retry'). Using last known: {self._map_name}")
            return self._map_name 
        
        _LOGGER.warning(f"Failed to get map name from vacuum after {counter} attempts. Result: {map_name_result}")
        return None 

    def _set_map_data(self, map_data: MapData):
        """Sets the internal map data and generates the PNG image."""
        if map_data and map_data.image and map_data.image.data:
            try:
                img_byte_arr = io.BytesIO()
                map_data.image.data.save(img_byte_arr, format='PNG')
                self._image = img_byte_arr.getvalue()
                self._store_image_if_enabled() 
            except Exception as e:
                _LOGGER.error(f"Error saving map image to byte array: {e}", exc_info=True)
                empty_map_obj = MapDataParser.create_empty(self._colors, "Image Error")
                img_byte_arr = io.BytesIO()
                empty_map_obj.image.data.save(img_byte_arr, format='PNG')
                self._image = img_byte_arr.getvalue()
        else:
            _LOGGER.warning("Attempted to set map data with invalid or missing image data. Creating empty map image.")
            current_status_msg = str(self._status) if self._status else "No Map Data"
            empty_map_obj = MapDataParser.create_empty(self._colors, current_status_msg)
            if empty_map_obj.image and empty_map_obj.image.data:
                img_byte_arr = io.BytesIO()
                empty_map_obj.image.data.save(img_byte_arr, format='PNG')
                self._image = img_byte_arr.getvalue()
            else: 
                self._image = None 

        self._map_data = map_data 

    def _create_device(self, user_id: Optional[str], device_id: Optional[str], model: str, mac: Optional[str]) -> XiaomiCloudVacuum:
        """Creates the appropriate cloud vacuum wrapper based on device model."""
        if not user_id or not device_id : 
            _LOGGER.error(f"Cannot create device wrapper for model {model}: UserID or DeviceID is missing.")
            return UnsupportedVacuum(self._connector, self._country, "UNKNOWN_USER", "UNKNOWN_DEVICE", model)

        self._used_api = self._detect_api(model)
        _LOGGER.info(f"Detected API for model {model}: {self._used_api}")

        if self._used_api == CONF_AVAILABLE_API_XIAOMI:
            return XiaomiVacuum(self._connector, self._country, user_id, device_id, model)
        if self._used_api == CONF_AVAILABLE_API_VIOMI:
            return ViomiVacuum(self._connector, self._country, user_id, device_id, model)
        if self._used_api == CONF_AVAILABLE_API_ROIDMI:
            return RoidmiVacuum(self._connector, self._country, user_id, device_id, model)
        if self._used_api == CONF_AVAILABLE_API_DREAME:
            return DreameVacuum(self._connector, self._country, user_id, device_id, model)
        if self._used_api == CONF_AVAILABLE_API_IJAI:
            if not mac:
                _LOGGER.error(f"Ijai API selected for model {model} but MAC address is missing. Map decryption will likely fail.")
            return IjaiVacuum(self._connector, self._country, user_id, device_id, model, self._token, self._host, mac)
        
        _LOGGER.warning(f"Unsupported API or model: {model}. Using default unsupported vacuum wrapper.")
        return UnsupportedVacuum(self._connector, self._country, user_id, device_id, model)

    def _detect_api(self, model: str) -> Optional[str]:
        """Detects the API to use based on the vacuum model."""
        if self._forced_api is not None:
            _LOGGER.debug(f"Using forced API: {self._forced_api} for model {model}")
            return self._forced_api
        if model in API_EXCEPTIONS:
            _LOGGER.debug(f"Model {model} found in API exceptions, using: {API_EXCEPTIONS[model]}")
            return API_EXCEPTIONS[model]

        for api, prefixes in AVAILABLE_APIS.items():
            for prefix in prefixes:
                if model.startswith(prefix):
                    _LOGGER.debug(f"Model {model} matches prefix {prefix}, using API: {api}")
                    return api
        
        _LOGGER.warning(f"Could not detect API for model: {model}. It might be unsupported.")
        return None 

    def _store_image_if_enabled(self):
        """Saves the current map image to disk if configured."""
        if self._store_map_image and self._image and self._device and self._device.model and self._store_map_path:
            try:
                if not os.path.exists(self._store_map_path):
                    _LOGGER.debug(f"Creating directory for storing map image: {self._store_map_path}")
                    os.makedirs(self._store_map_path, exist_ok=True)
                
                image_path = f"{self._store_map_path}/map_image_{self._device.model}.png"
                with open(image_path, "wb") as f:
                    f.write(self._image)
                _LOGGER.debug(f"Saved map image to {image_path}")
            except Exception as e:
                _LOGGER.warning(f"Error while saving map image to {self._store_map_path}: {e}", exc_info=True)


class CameraStatus(Enum):
    """Status messages for the camera."""
    EMPTY_MAP = 'Retrieved map is empty'
    FAILED_LOGIN = 'Failed to log in to Xiaomi Cloud'
    FAILED_TO_RETRIEVE_DEVICE = 'Failed to retrieve device details from Xiaomi Cloud'
    FAILED_TO_RETRIEVE_MAP_FROM_VACUUM = 'Failed to retrieve map name from vacuum (device)'
    INITIALIZING = 'Initializing'
    LOGGED_IN = 'Logged in to Xiaomi Cloud'
    NOT_LOGGED_IN = 'Not logged in to Xiaomi Cloud'
    OK = 'Map retrieved successfully' # This status will also be used when reusing in-memory map
    TWO_FACTOR_AUTH_REQUIRED = 'Two-factor authentication required (check HA logs/attributes for URL)'
    UNABLE_TO_PARSE_MAP = 'Unable to parse retrieved map data'
    UNABLE_TO_RETRIEVE_MAP = 'Unable to retrieve map data from Xiaomi Cloud' 

    def __str__(self):
        return str(self._value_)

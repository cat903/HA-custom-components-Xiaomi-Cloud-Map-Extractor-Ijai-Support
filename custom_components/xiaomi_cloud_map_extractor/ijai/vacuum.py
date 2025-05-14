from typing import Optional
from custom_components.xiaomi_cloud_map_extractor.ijai.map_data_parser import MapDataParserIjai
from miio.miot_device import MiotDevice

from custom_components.xiaomi_cloud_map_extractor.common.vacuum_v2 import XiaomiCloudVacuumV2
from custom_components.xiaomi_cloud_map_extractor.types import Colors, Drawables, ImageConfig, Sizes, Texts
from custom_components.xiaomi_cloud_map_extractor.common.map_data import MapData

import logging
_LOGGER = logging.getLogger(__name__)

class IjaiVacuum(XiaomiCloudVacuumV2):
    WIFI_STR_LEN = 18
    WIFI_STR_POS = 11

    def __init__(self, connector, country, user_id, device_id, model, token, host, mac):
        super().__init__(connector, country, user_id, device_id, model)
        self._token = token
        self._host = host
        self._mac = mac
        self._wifi_info_sn = None

    def get_map_archive_extension(self) -> str:
        return "zlib"

    def get_map_url(self, map_name: str) -> Optional[str]: # Now Optional is defined
        url = self._connector.get_api_url(self._country) + '/v2/home/get_interim_file_url_pro'
        params = {
            "data": f'{{"obj_name":"{self._user_id}/{self._device_id}/{map_name}"}}'
        }
        api_response = self._connector.execute_api_call_encrypted(url, params)
        if api_response is None or ("result" not in api_response) or \
           (api_response["result"] is None) or ("url" not in api_response["result"]):
            code = api_response.get('code', 'N/A') if api_response else 'N/A'
            message = api_response.get('message', 'No response or unexpected structure') if api_response else 'No response'
            _LOGGER.debug(f"API returned {code}({message}) when trying to get map URL for {map_name}")
            return None
        return api_response["result"]["url"]

    def decode_map(self,
                   raw_map: bytes,
                   colors: Colors,
                   drawables: Drawables,
                   texts: Texts,
                   sizes: Sizes,
                   image_config: ImageConfig) -> Optional[MapData]: # Now Optional is defined
        GET_PROP_RETRIES = 5
        if self._wifi_info_sn is None or self._wifi_info_sn == "":
            _LOGGER.debug(f"Attempting to get wifi_sn from vacuum: host={self._host}, token_len={len(self._token) if self._token else 0}")
            # Retry loop for getting wifi_info_sn
            for attempt in range(GET_PROP_RETRIES):
                try:
                    # Ensure MiotDevice is compatible with your miio library version
                    # lazy_discover might need to be False for some setups for immediate connection.
                    device = MiotDevice(self._host, self._token, lazy_discover=True) # Try with lazy_discover=True first
                    
                    # The specific siid and piid for wifi_info_sn are crucial and model-dependent.
                    # Example: (siid=7, piid=45) was used previously. Verify this for Ijai v18.
                    # Some devices might expose this under a different property.
                    props_result = device.get_property_by(siid=7, piid=45) # Check your device's MIoT spec
                    
                    if props_result and isinstance(props_result, list) and len(props_result) > 0 and "value" in props_result[0]:
                        props_str = props_result[0]["value"]
                        if isinstance(props_str, str):
                            props = props_str.split(',')
                            if len(props) > self.WIFI_STR_POS:
                                self._wifi_info_sn = props[self.WIFI_STR_POS].replace('"', '')[:self.WIFI_STR_LEN]
                                _LOGGER.debug(f"Successfully retrieved wifi_sn: {self._wifi_info_sn} on attempt {attempt + 1}")
                                break # Exit retry loop on success
                            else:
                                _LOGGER.warning(f"wifi_info_sn string from MIoT property not in expected format or too short on attempt {attempt + 1}. Raw: {props_str}")
                        else:
                            _LOGGER.warning(f"MIoT property for wifi_info_sn did not return a string on attempt {attempt + 1}. Received: {props_str}")
                    else:
                        _LOGGER.warning(f"Could not retrieve MIoT property for wifi_info_sn on attempt {attempt + 1}. Result: {props_result}")
                
                except Exception as e:
                    _LOGGER.warning(f"Attempt {attempt + 1}/{GET_PROP_RETRIES} to get wifi_sn from vacuum {self._host} failed: {e}", exc_info=True if attempt == GET_PROP_RETRIES -1 else False)
                    if attempt < GET_PROP_RETRIES - 1:
                        time.sleep(1) # Wait a bit before retrying
                    else: # Last attempt failed
                         _LOGGER.error(f"Exhausted retries for getting wifi_sn from {self._host}.")
                if self._wifi_info_sn: # If successfully fetched in any attempt
                    break


        if self._wifi_info_sn is None: # Check after retries
             _LOGGER.error("wifi_info_sn is None after all retries. Map decryption for Ijai vacuum will likely fail if wifi_sn is required.")
             # Depending on whether wifi_info_sn is strictly required for all Ijai maps or only some:
             # return None # Option 1: Fail early if wifi_info_sn is always needed.
             # Option 2: Proceed and let unpack_map handle it (it might work for some maps or if it's optional)
             _LOGGER.warning("Proceeding with map decryption without wifi_info_sn. This may fail.")


        try:
            decrypted_map_bytes = MapDataParserIjai.unpack_map(
                raw_map,
                wifi_sn=self._wifi_info_sn if self._wifi_info_sn else "", 
                owner_id=str(self._user_id),
                device_id=str(self._device_id),
                model=self.model, 
                device_mac=self._mac if self._mac else "" 
            )
            if decrypted_map_bytes is None:
                _LOGGER.error("Failed to unpack/decrypt Ijai map.")
                return None
            
            return MapDataParserIjai.parse(decrypted_map_bytes, colors, drawables, texts, sizes, image_config)
        except Exception as e:
            _LOGGER.error(f"Error during Ijai map decoding or parsing after potential decryption: {e}", exc_info=True)
            return None

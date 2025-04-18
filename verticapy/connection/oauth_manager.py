"""
Copyright  (c)  2018-2025 Open Text  or  one  of its
affiliates.  Licensed  under  the   Apache  License,
Version 2.0 (the  "License"); You  may  not use this
file except in compliance with the License.

You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Unless  required  by applicable  law or  agreed to in
writing, software  distributed  under the  License is
distributed on an  "AS IS" BASIS,  WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
See the  License for the specific  language governing
permissions and limitations under the License.
"""

from __future__ import print_function, division, absolute_import, annotations

import requests

from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

from verticapy._utils._print import print_message
from verticapy.connection.errors import (
    OAuthTokenRefreshError,
    OAuthConfigurationError,
    OAuthEndpointDiscoveryError,
)


# Never check any hostnames
class HostNameIgnoringAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(
            num_pools=connections, maxsize=maxsize, block=block, assert_hostname=False
        )


class OAuthManager:
    def __init__(self, refresh_token):
        self.refresh_token = refresh_token
        self.client_id = ""
        self.client_secret = ""
        self.token_url = ""
        self.discovery_url = ""
        self.scope = ""
        self.tls_check_hostname = True
        self.tls_verify = True
        self.refresh_attempted = False

    def set_config(self, configs) -> None:
        valid_keys = {
            "refresh_token",
            "client_id",
            "client_secret",
            "token_url",
            "discovery_url",
            "scope",
            "tls_check_hostname",
            "tls_verify",
        }
        try:
            for k, v in configs.items():
                if k not in valid_keys:
                    invalid_key = f"Unrecognized OAuth config property: {k}"
                    print_message(invalid_key, "warning")
                    continue
                if v is None or v == "":  # ignore empty value
                    continue
                if k == "refresh_token":
                    self.refresh_token = str(v)
                elif k == "client_id":
                    self.client_id = str(v)
                elif k == "client_secret":
                    self.client_secret = str(v)
                elif k == "token_url":
                    self.token_url = str(v)
                elif k == "discovery_url":
                    self.discovery_url = str(v)
                elif k == "scope":
                    self.scope = str(v)
                elif k == "tls_check_hostname":
                    self.tls_check_hostname = bool(v)
                elif k == "tls_verify":
                    if not isinstance(v, (bool, str)):
                        raise ValueError('"tls_verify" should be a bool or str')
                    self.tls_verify = v  # bool or str
        except Exception as e:
            raise OAuthConfigurationError("Failed setting OAuth configuration.") from e

    def get_access_token_using_refresh_token(self):
        """Issue a new access token using a valid refresh token."""
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Expires": "0",
        }
        params = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
        }
        if self.scope:
            params["scope"] = self.scope
        err_msg = "Failed getting OAuth access token from a refresh token."
        try:
            s = requests.Session()
            if not self.tls_check_hostname:
                s.mount("https://", HostNameIgnoringAdapter())
            response = s.post(
                self.token_url, headers=headers, data=params, verify=self.tls_verify
            )
            response.raise_for_status()
            json_response = response.json()
            # If refresh token rotation is used, like in OTDS, we will get both our new valid access token as well as
            # a new refresh token to use the next time we need to invoke token refresh.
            if "refresh_token" in json_response:
                self.refresh_token = json_response["refresh_token"]
            return response.json()["access_token"]
        except requests.exceptions.HTTPError as err:
            msg = f"{err_msg}\n{err}\n{response.json()}"
            raise OAuthTokenRefreshError(msg)
        except Exception as e:
            raise OAuthTokenRefreshError(err_msg) from e

    def _get_token_url_from_discovery_url(self) -> str:
        try:
            headers = {
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Expires": "0",
            }
            s = requests.Session()
            if not self.tls_check_hostname:
                s.mount("https://", HostNameIgnoringAdapter())
            response = s.get(
                self.discovery_url, headers=headers, verify=self.tls_verify
            )
            response.raise_for_status()
            return response.json()["token_endpoint"]
        except Exception as e:
            err_msg = "Failed getting token url from discovery url."
            raise OAuthEndpointDiscoveryError(err_msg) from e

    def do_token_refresh(self) -> str:
        self.refresh_attempted = True

        if len(self.token_url) == 0 and len(self.discovery_url) == 0:
            raise OAuthTokenRefreshError("Token URL or Discovery URL must be set.")
        if len(self.client_id) == 0:
            raise OAuthTokenRefreshError("OAuth client id is missing.")
        if len(self.refresh_token) == 0:
            raise OAuthTokenRefreshError("OAuth refresh token is missing.")
        # client_secret is not required for non confidential clients

        # If the token url is not set, get it from the discovery url
        if len(self.token_url) == 0:
            self.token_url = self._get_token_url_from_discovery_url()

        return self.get_access_token_using_refresh_token()

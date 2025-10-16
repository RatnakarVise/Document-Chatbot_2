import io
import base64
import requests


def get_graph_access_token(tenant_id: str, client_id: str, client_secret: str) -> str:
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    token_data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://graph.microsoft.com/.default"
    }
    resp = requests.post(token_url, data=token_data)
    resp.raise_for_status()
    return resp.json()["access_token"]


def detect_link_type(url: str) -> str:
    # Returns "folder_share", "file_share", or "site_path"
    if "/:f:/" in url:
        return "folder_share"
    if "/:b:/" in url or "/:w:/" in url:
        return "file_share"
    return "site_path"


def _encode_share_url(url: str) -> str:
    return base64.urlsafe_b64encode(url.strip().encode("utf-8")).decode("utf-8").rstrip("=")


def fetch_folder_items_from_share_link(folder_share_url: str, access_token: str):
    encoded = _encode_share_url(folder_share_url)
    url = f"https://graph.microsoft.com/v1.0/shares/u!{encoded}/driveItem/children"
    res = requests.get(url, headers={"Authorization": f"Bearer {access_token}"})
    res.raise_for_status()
    return res.json().get("value", [])


def fetch_file_from_share_link(file_share_url: str, access_token: str, meta_only: bool = False):
    encoded = _encode_share_url(file_share_url)
    meta_url = f"https://graph.microsoft.com/v1.0/shares/u!{encoded}/driveItem"
    meta_res = requests.get(meta_url, headers={"Authorization": f"Bearer {access_token}"})
    meta_res.raise_for_status()
    meta = meta_res.json()
    if meta_only:
        return meta
    return meta


def resolve_site_and_list_folder(hostname: str, site_name: str, access_token: str) -> str:
    # Returns site id (sites/{site-id})
    api = f"https://graph.microsoft.com/v1.0/sites/{hostname}:/sites/{site_name}"
    res = requests.get(api, headers={"Authorization": f"Bearer {access_token}"})
    res.raise_for_status()
    return res.json()["id"]


def fetch_folder_items_with_site_id(site_id: str, folder_path: str, access_token: str):
    # Fetch one-level children of the folder; caller can recurse if needed
    url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root:/{folder_path}:/children"
    res = requests.get(url, headers={"Authorization": f"Bearer {access_token}"})
    res.raise_for_status()
    return res.json().get("value", [])
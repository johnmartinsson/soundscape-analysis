"""
Shows basic usage of the Drive v3 API.

Creates a Drive v3 API service and prints the names and ids of the last 10 files
the user has access to.
"""
from __future__ import print_function
from apiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools
from apiclient.http import MediaIoBaseDownload
import io
import sys
import os

def recursive_download(path_id, directory_path):
    print("Path id: ", path_id)
    print("directory_path: ", directory_path)
    should_break = False
    list_of_files = {'nextPageToken': None}
    while not should_break:
        list_of_files = filesresource.list(
            q="'{}' in parents".format(path_id),
            pageSize=1000,
            pageToken=list_of_files['nextPageToken']
        ).execute()

        if not ('nextPageToken' in list_of_files and list_of_files['nextPageToken'] != ''):
            should_break = True

            for file_dict in list_of_files["files"]:
                if file_dict["mimeType"] == "application/vnd.google-apps.folder":
                    recursive_download(file_dict['id'], os.path.join(directory_path, file_dict['name']))
                else:
                    file_id = file_dict["id"]
                    filename = file_dict["name"]

                    print("Downloading %s"%filename)
                    request = filesresource.get_media(fileId=file_id)
                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)
                    fh = open(os.path.join(directory_path, filename), "wb")
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()
                        print("Download %d%%." % int(status.progress() * 100))
                    fh.close()

if __name__ == "__main__":
    # Setup the Drive v3 API
    SCOPES = 'https://www.googleapis.com/auth/drive.readonly'
    store = file.Storage('credentials.json')
    flow = client.flow_from_clientsecrets('client_secret.json', SCOPES)
    creds = tools.run_flow(flow, store)
    service = build('drive', 'v3', http=creds.authorize(Http()))
    filesresource = service.files()

    path_id = input("Google Drive folder id: ")

    recursive_download(path_id, "/mnt/storage_1/john/data/madagascar/SM1/")

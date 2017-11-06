import urllib
import urllib.request
import json
import jsonpath
import os
import shutil

#https://c-w.github.io/gutenberg-http/

remove_corpus = True

def main(args=None):
    authors = []
    authors.append("Goethe, Johann Wolfgang von")
    authors.append("Schiller, Friedrich")
    authors.append("Lessing, Gotthold Ephraim")
    authors.append("Hesse, Hermann")
    corpus_path = "corpus"
    download_works_by_authors(authors, corpus_path)

def download_works_by_authors(authors, corpus_path="corpus"):

    # Make sure that there is a proper corpus-folder.
    if os.path.exists(corpus_path) and remove_corpus is True:
        shutil.rmtree(corpus_path)
    if not os.path.exists(corpus_path):
        os.makedirs(corpus_path)

    for author in authors:
        print("Downloading all works of author", author)

        # Create a folder for the autor.
        text_collection_path = os.path.join(corpus_path, author)
        if not os.path.exists(text_collection_path):
            os.makedirs(text_collection_path)

        # Generate URL for querying all of the author's texts.
        list_of_works_url_string = "author eq " + author + "and language eq de"
        list_of_works_url_string = urllib.parse.quote(list_of_works_url_string)
        list_of_works_url_string = "https://gutenbergapi.org/search/" + list_of_works_url_string
        print("URL", list_of_works_url_string)

        # Get all text-ids.
        with urllib.request.urlopen(list_of_works_url_string) as list_of_works_url:
            json_data = json.loads(list_of_works_url.read().decode())

            # Get text-ids.
            text_ids = jsonpath.jsonpath(json_data, '$..text_id')
            print("Loading", len(text_ids), "texts...")

            # TODO consider progress bar
            for text_id in text_ids:
                print("Downloading text with id", text_id)

                # Load the metadata. TODO Maybe remove.
                #text_metadata_url_string = "https://gutenbergapi.org/texts/" + str(text_id)
                #print("URL", text_metadata_url_string)
                #text_metadata_file_name = str(text_id) + ".metadata.json.txt"
                #text_metadata_file_path = os.path.join(text_collection_path, text_metadata_file_name)
                #print("File", text_metadata_file_path)
                #if not os.path.exists(text_metadata_file_path):
                #    with urllib.request.urlopen(text_metadata_url_string) as text_metadata_url:
                #        text_metadata = json.loads(text_metadata_url.read().decode())
                #        text_metadata = json.dumps(text_metadata, indent=4, sort_keys=True)
                #        with open(text_metadata_file_path, "w") as text_file:
                #            text_file.write(text_metadata)
                            # TODO Consider pretty print.

                # Load the body.
                text_body_url_string = "https://gutenbergapi.org/texts/" + str(text_id) + "/body"
                print("URL", text_body_url_string)
                text_body_file_name = str(text_id) + ".body.txt"
                text_body_file_path = os.path.join(text_collection_path, text_body_file_name)
                print("File", text_body_file_path)
                if not os.path.exists(text_body_file_path):
                    with urllib.request.urlopen(text_body_url_string) as text_body_url:
                        text_body = json.loads(text_body_url.read().decode())
                        text_body = text_body["body"]
                        text_body = clean_up_body(text_body)
                        with open(text_body_file_path, "w") as text_file:
                            text_file.write(text_body)

def clean_up_body(text_body):

    # TODO Comment variant 1
    start_string =  "START OF"
    start_string_index = text_body.find(start_string)
    end_string = "END OF"
    end_string_index = text_body.find(end_string)
    if start_string_index != -1:
        text_body = text_body[start_string_index:end_string_index]
        text_body = text_body[text_body.find("***") + 3:]
        return text_body

    # TODO Comment variant 2
    small_print_end_string = "*END*"
    small_print_end_string_index = text_body.rfind(small_print_end_string)
    if small_print_end_string_index != -1:
        text_body = text_body[small_print_end_string_index + len(small_print_end_string):]
        return text_body

    raise Exception("Text could not be cleaned up.")


if __name__ == "__main__":
    main()

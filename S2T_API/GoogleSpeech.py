#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API using the REST API for async
batch processing.

How to use:
    python GoogleSpeech.py Filename.flac num_speakers
    python GoogleSpeech.py gs://cloud-samples-tests/speech/vr.flac num_speakers

Base code from github.com/googleapis
Adjustments by Sophia & Jeff
"""

import argparse
import io
import os
import json
import pickle
import pandas as pd

from google.cloud import speech

# TODO: Enable data logging for price discount; (Data logging is disabled for this project for Google Cloud Speech API)
#       Speech Adaptation: https://cloud.google.com/speech-to-text/docs/context-strength, https://cloud.google.com/speech-to-text/docs/speech-adaptation

# TODO: add in compatibility with phrase hints, saving results to bucket, passing in multiple files, saving output as text file.


# [START speech_transcribe_async_gcs]
def transcribe_gcs(gcs_uri, output_dir):
    """Asynchronously transcribes the audio file specified by the gcs_uri."""

    # Imports the Google Cloud client library
    #from google.cloud import speech

    # Instantiates a client
    client = speech.SpeechClient()

    # Construct a recognition metadata object
    metadata = speech.RecognitionMetadata()
    metadata.interaction_type = speech.RecognitionMetadata.InteractionType.DISCUSSION
    metadata.recording_device_type = (
        speech.RecognitionMetadata.RecordingDeviceType.OTHER_INDOOR_DEVICE
    )
    metadata.audio_topic = "court trial hearing"
    metadata.original_mime_type = "audio/mp3"

    audio = speech.RecognitionAudio(uri=gcs_uri)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
        enable_word_confidence=True,
        # Enhanced models cost more than standard models.
        use_enhanced=True,
        model="video",
        enable_word_time_offsets=True,

    )

    # Detects speech in the audio file -- long audio file
    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    response = operation.result()
    print(type(response))

    print("Saving down results")
    # Writing results to json
    output_json = {}
    results_counter = 0
    print(len(response.results))

    # with open('../../sample.pickle', 'wb') as fh:
    #     pickle.dump(response, fh)

    filename = gcs_uri.rsplit('/', 1)[-1]
    output_text = "*" * 100
    output_text += f'\n{filename:^100s}\n'
    output_text += "*" * 100
    output_text += '\n'

    for ii, result in enumerate(response.results):
        alternative = result.alternatives[0]
        conf = result.alternatives[0].confidence
        if len(result.alternatives) > 1:
            for alt in result.alternatives:
                if alt.confidence > conf:
                    alternative = alt
                    conf = alt.confidence


        by_word = []
        for word_info in alternative.words:
            by_word.append([word_info.word,
                            word_info.start_time.total_seconds(),
                            word_info.end_time.total_seconds()])

        transcript = alternative.transcript

        if len(by_word) > 0:
            start_time = by_word[0][1]
            end_time = by_word[-1][-1]

        output_text += f"\nStart Time: {start_time:5.3f}s\t End Time: {end_time:5.3f}s\t Confidence: {conf*100:.4f}%\n"
        output_text += f"Transcript: {transcript}\n"

        output_json[results_counter]=[transcript, conf, by_word]
        results_counter += 1


    if not os.path.exists(output_dir):
        print(f"Creating new output_dir at {output_dir}!\n")
        os.makedirs(output_dir)

    json_savepath = os.path.join(output_dir, f"{filename.split('.')[0]}.json")
    with open(f"{json_savepath}", "w+") as file:
        json.dump(output_json, file)

    text_savepath = os.path.join(output_dir, f"{filename.split('.')[0]}.txt")
    pd.DataFrame([output_text]).to_csv(text_savepath, index=False, header=False)

    print(f"Diarized and transcribed {filename}. Output saved in {output_dir} Dir!")

# [END speech_transcribe_async_gcs]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'input_path', help='File or GCS path for audio file to be recognized')
    parser.add_argument(
        'output_dir', help='Dir where the results will be saved')
    #parser.add_argument('num_speakers', help='diarization_speaker_count')
    args = parser.parse_args()

    transcribe_gcs(args.input_path, args.output_dir)

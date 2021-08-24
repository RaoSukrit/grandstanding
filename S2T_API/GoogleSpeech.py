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

# TODO: Enable data logging for price discount; (Data logging is disabled for this project for Google Cloud Speech API) 
#       Speech Adaptation: https://cloud.google.com/speech-to-text/docs/context-strength, https://cloud.google.com/speech-to-text/docs/speech-adaptation

# [START speech_transcribe_async_gcs]
def transcribe_gcs(gcs_uri):
    """Asynchronously transcribes the audio file specified by the gcs_uri."""

    # Imports the Google Cloud client library
    #from google.cloud import speech
    from google.cloud import speech

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
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
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
    for result in response.results:
        temp_dict = {}
        alt = result.alternatives[0]
        trans = alt.transcript
        conf = alt.confidence
        by_word = []
        for word_info in alt.words:
            by_word.append([word_info.word, word_info.start_time.total_seconds(), word_info.end_time.total_seconds()])
        output_json[results_counter]=[trans, conf, by_word]
        results_counter+=1
            
        
    with open("{}.json".format(gcs_uri.split('/')[-1][:-5]) , "w+") as file:
        json.dump(output_json, file)
    
    print("Diarized and transcribed {}".format(gcs_uri.split('/')[-1]))

# [END speech_transcribe_async_gcs]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'path', help='File or GCS path for audio file to be recognized')
    #parser.add_argument('num_speakers', help='diarization_speaker_count')
    args = parser.parse_args()

    transcribe_gcs(args.path)

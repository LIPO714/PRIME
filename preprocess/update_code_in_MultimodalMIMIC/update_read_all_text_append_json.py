'''
Update funtion "read_all_text_append_json" in text_util.py by following code:
'''

    def read_all_text_append_json(self, names, period_length=48.0, NumOfNotes=5, notes_aggeregate='Mean'):
        texts_dict = {}
        time_dict = {}
        start_times = {}
        for patient_id in names:
            pid, eid = self.get_name_from_filename(patient_id)
            text_file_name = pid + "_" + eid
            if text_file_name in self.all_files:
                # print(text_file_name)
                time, texts = self.read_text_event_json(text_file_name)
                start_time = self.episodeToStartTime[text_file_name]
                if len(texts) == 0 or start_time == -1:
                    continue
                final_concatenated_text = []
                times_array = []
                for (t, txt) in zip(time, texts):

                    if period_length is not None:  # 48ihm task专用
                        if diff(start_time, t) <= period_length + 1e-6 : #and  diff(start_time, t)>=(-24-1e-6)
                            final_concatenated_text.append(txt)
                            times_array.append(t)
                        else:
                            break
                    else:  # pheno task 都存下来
                        final_concatenated_text.append(txt)
                        times_array.append(t)

                if len(final_concatenated_text)==0:
                    continue
                texts_dict[patient_id] = final_concatenated_text
                time_dict[patient_id] = times_array
                start_times[patient_id] = start_time
        return texts_dict, time_dict, start_times

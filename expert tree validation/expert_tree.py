""""
New expert tree that was based on the changes made on 10.19.2020 with Dr Fu
"""

SPECIAL = -1
non_LE = 0
Latent = 1
Mild = 2
Moderate = 3
Severe = 4

# last update update:10.21.2020


class Expert_Tree:
    def __init__(self, row, is_Kinect=False):
        namemap = {'mobility': 'Mobility', 'AS': 'ArmSwelling', 'BS': 'BreastSwelling',
                   'skin': 'Skin', 'pas': 'PAS', 'fht': 'FHT', 'discomfort': 'DISCOMFORT',
                   'symcount': 'SYM_COUNT', 'timelap': 'TIME_LAPSE', 'LVC': 'LVC',
                   'CS': 'ChestWallSwelling', 'fluid': 'fluid_total'}
        self.d = {key: row[namemap[key]] for key in namemap}
        self.str = ''
        self.is_Kinect = is_Kinect
#         self.d['mobility']=  row[0]
#         self.d['AS']= row[1]
#         self.d['BS']= row[2]
#         self.d['skin']= row[3]
#         self.d['pas']= row[4]
#         self.d['fht']= row[5]
#         self.d['discomfort']= row[6]
#         self.d['symcount']= row[7]
#         self.d['timelap']= row[8]
#         ##??? fluid question
#         #self.d['fluid'] = max(self.d['AS'], self.d['BS'],self.d['pas'],self.d['fht'],self.d['skin'] )#???
#         self.d['fluid'] = row[11]
#         self.d['LVC'] = row[9]
#         self.d['CS'] = rrow[10]
#        # self.d['fluid'] = row[11]
#         self.str = ''

    def run(self):
        # change based on Kinect
        if (self.is_Kinect and self.d['LVC'] <= 7.1) or (not self.is_Kinect and self.d['LVC'] <= 0.05):
            if self.d['AS'] == 0:
                if self.d['BS'] == 0:
                    if self.d['symcount'] > 8 and self.d['timelap'] > 0.5:
                        self.str += 'a'
                        return Mild

                elif self.d['BS'] == 1:
                    if self.d['skin'] > 1 and self.d['fht'] > 0:
                        self.str += 'b'
                        return Mild
                    if self.d['skin'] <= 1 or self.d['fht'] == 0:
                        if self.d['CS'] > 1 and self.d['timelap'] >= 1:
                            self.str += 'c'
                            return Mild
                        elif self.d['CS'] <= 1:  # ask
                            # *************** 12 month change
                            if self.d['discomfort'] > 1 and self.d['timelap'] >= 1:
                                self.str += 'd'
                                return Mild
                            # new change here
                            elif self.d['mobility'] >= 1 and self.d['timelap'] < 1:
                                self.str += 'e'
                                return Mild
                            elif self.d['pas'] > 1 and self.d['timelap'] > 0.5:
                                self.str += 'f'
                                return Mild

                    if self.d['symcount'] > 8 and self.d['timelap'] > 0.5:
                        self.str += 'g'
                        return Mild
                else:  # BS>1
                    if self.d['skin'] > 1:
                        self.str += 'h'
                        return Mild

                    if self.d['skin'] <= 1:  # skin<=1
                        if self.d['timelap'] >= 1:
                            self.str += 'i'
                            return Mild
                        elif self.d['mobility'] >= 1 and self.d['timelap'] > 0.5:  # ask? mobility?
                            self.str += 'j'
                            return Mild
                         # ^^^^ timlap<=6 months condition removed:
                    if self.d['symcount'] > 8 and self.d['timelap'] > 0.5:
                        self.str += 'k'
                        return Mild

            elif self.d['AS'] >= 1:
                if self.d['skin'] == 0:
                    if self.d['symcount'] <= 8:
                        self.str += 'l'
                        return Mild
                    else:  # symptom >8
                        self.str += 'm'
                        return Moderate
                elif self.d['skin'] == 1:
                    self.str += 'n'
                    return Moderate
                elif self.d['skin'] > 1:
                    if self.d['fht'] > 0 and self.d['pas'] > 0 and self.d['mobility'] > 0 and self.d['symcount'] > 14:
                        self.str += 'o'
                        return Severe
                    else:
                        self.str += 'q'
                        return Moderate

            #  else:  NONE OF ABOVE
            # still check here to catch any patients that are not caught
            if (self.is_Kinect and self.d['LVC'] > 5) or (not self.is_Kinect and self.d['LVC'] > 0.03) and self.d['symcount'] > 8 and self.d['timelap'] > 0.5:
                self.str += 'r'
                return Mild
            else:
                self.str += 's'
                return Latent

        else:  # LVC >0.05 or LDEX>7.1
            # AS ==0
            if self.d['AS'] == 0:
                if self.d['BS'] == 0:
                    if self.d['CS'] > 1 and self.d['fht'] > 0:
                        if self.d['pas'] == 0:
                            self.str += 't'
                            return Mild
                        else:  # pas>0
                            self.str += 'u'
                            return Severe

                    if self.d['symcount'] <= 8:
                        self.str += 'x'
                        return Latent
                    # ASK HERE  DOUBLE CHECK THE BOUNDARY
                    elif 8 < self.d['symcount'] and self.d['symcount'] <= 14:
                        if self.d['timelap'] <= 0.5:
                            self.str += 'y'
                            return Latent
                        else:
                            self.str += 'z'
                            return Mild
                    else:  # symp_count >14
                        self.str += 'A'
                        return Moderate
                # BS >= 1
                elif self.d['BS'] >= 1:
                    if self.d['mobility'] > 1:
                        self.str += 'B'
                        return Moderate
                    if self.d['skin'] > 1:
                        self.str += 'C'
                        return Moderate
                    if self.d['discomfort'] > 1:
                        self.str += 'D'
                        return Moderate
                    if self.d['BS'] > 1:
                        self.str += 'E'
                        return Moderate
                    if self.d['mobility'] == 1:
                        if self.d['timelap'] <= 0.5:
                            self.str += 'F'
                            return Latent
                        else:  # self.d['timelap'] > 0.5:
                            self.str += 'G'
                            return Mild

            elif self.d['AS'] >= 1:
                if self.d['fht'] > 1 or self.d['pas'] > 1:
                    self.str += 'I'
                    return Severe
                elif self.d['mobility'] > 1:
                    self.str += 'J'
                    return Severe
                elif self.d['skin'] > 1:
                    self.str += 'K'
                    return Severe
                elif self.d['discomfort'] > 1:
                    self.str += 'L'
                    return Severe
                # if none of above
                if self.d['BS'] > 1:
                    return Moderate
                if self.d['mobility'] == 1 and self.d['timelap'] > 0.5:
                    self.str += 'M'
                    return Moderate
                if (self.d['fht'] >= 1 or self.d['pas'] >= 1) and self.d['timelap'] > 0.5:
                    self.str += 'N'
                    return Moderate

            # if none above
            if self.d['symcount'] > 8 and self.d['timelap'] > 0.5:
                self.str += 'O'
                return Mild
            else:
                self.str += 'P'
                return Latent
        self.str += 'X'
        return SPECIAL
        # return non_LE


def get_expert_tree_results(data, is_Kinect, class_number):
    flags = []
    # data_lst = list(X_arr)
    # row,col = data.shape[0], data.shape[1]
    # print((row,col))
    # print(len(LVC_lst))
    expert_tree_result = []
    expert_truth_labels = []

    for i, row in data.iterrows():  # row = 101
        # in row
        #         row_lst = []
        #         for c in range(0,col):
        #             row_lst.append(data[r][c])

        # run the algorithm in the expert tree
        exp_tree = Expert_Tree(row, is_Kinect)
        expert_tag = exp_tree.run()

        """
        non_LE= 0
        Latent = 1
        Mild = 2
        Moderate = 3
        Severe = 4

        expand expert_truth_labels_accordingly
        """
        if expert_tag == 0:
            expert_truth_labels.append('Non_LE')
        elif expert_tag == 1:
            expert_truth_labels.append('Latent')
        elif expert_tag == 2:
            expert_truth_labels.append('Mild')
        elif expert_tag == 3:
            expert_truth_labels.append('Moderate')
        elif expert_tag == 4:
            expert_truth_labels.append('Severe')
        else:
            expert_truth_labels.append('ERROR!')

        if class_number == 3:
            if expert_tag == 1 or expert_tag == 0:  # none or latent
                result = 0
            elif expert_tag == 2:  # mild
                result = 1
            elif expert_tag > 2:  # moderate severe
                result = 2
            else:
                result = -999
        elif class_number == 2:  # if not tree clas (if it two-class)
            if expert_tag == 1 or expert_tag == 0:  # none or latent
                result = 0
            elif expert_tag > 1:  # mild moderate severe
                result = 1
            else:
                result = -999
        elif class_number == 4:  # class number 4
            if expert_tag == 1 or expert_tag == 0:  # none or latent
                result = 0
            elif expert_tag == 2:  # mild
                result = 1
            elif expert_tag == 3:  # moderate
                result = 3
            elif expert_tag == 4:  # severe
                result = 4
            else:
                result = -999
        else:
            print('invalid class number')

        expert_tree_result.append(result)
        flags.append(exp_tree.str)

    return expert_tree_result, expert_truth_labels, flags

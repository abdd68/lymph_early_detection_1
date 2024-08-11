""""
New expert tree that was based on the changes made on 4.28.2020 with Dr Fu
"""

SPECIAL = -1
non_LE = 0
Latent = 1
Mild = 2
Moderate = 3
Severe = 4

# update: 3.3.2020 00.26 am
# code with all 12 fixed to original thesis model,everything in that drawing


class Expert_Tree:
    def __init__(self, row):
        namemap = {'mobility': 'Mobility', 'AS': 'ArmSwelling', 'BS': 'BreastSwelling',
                   'skin': 'Skin', 'pas': 'PAS', 'fht': 'FHT', 'discomfort': 'DISCOMFORT',
                   'symcount': 'SYM_COUNT', 'timelap': 'TIME_LAPSE', 'LVC': 'LVC',
                   'CS': 'ChestWallSwelling', 'fluid': 'fluid_total'}
        self.d = {key: row[namemap[key]] for key in namemap}
        self.str = ''
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
        if self.d['LVC'] <= 7.1:  # CHANGED THIS COLUMN FOR KINECT LDEX VALUES
            if self.d['AS'] == 0:
                if self.d['BS'] == 0:
                    if self.d['skin'] > 1:
                        if self.d['symcount'] < 6:
                            self.str += 'a'
                            return Latent
                        else:
                            self.str += 'b'
                            return Mild
                    else:  # skin <=1
                        # *************** 12 month change
                        if self.d['CS'] > 1 and self.d['timelap'] >= 1:  # check month
                            self.str += 'c'
                            return Mild
                        elif self.d['CS'] <= 1:
                            if self.d['fht'] == 0:
                                if self.d['discomfort'] >= 1 and self.d['timelap'] <= 0.5:
                                    self.str += 'd'
                                    return Latent
                                if (self.d['mobility'] >= 1 or self.d['pas'] >= 1) and self.d['timelap'] <= 0.5:
                                    self.str += 'e'
                                    return Latent

                            elif self.d['fht'] >= 1:
                                # *************** 12 month change
                                if self.d['discomfort'] > 1 and self.d['timelap'] >= 1:
                                    self.str += 'f'
                                    return Mild
                                if self.d['mobility'] >= 1 and self.d['timelap'] <= 0.5:
                                    self.str += 'g'
                                    return Mild

                                if self.d['fht'] == 1 and self.d['pas'] > 1 and self.d['timelap'] > 0.5:
                                    self.str += 'h'
                                    return Mild
                            # None of above
                            # *************** 12 month change
                            if self.d['CS'] == 1 and self.d['timelap'] < 1:
                                self.str += 'i'
                                return Latent

                elif self.d['BS'] == 1:
                    if self.d['skin'] > 1 and self.d['fht'] > 0:
                        self.str += 'j'
                        return Moderate
                    elif self.d['skin'] <= 1 or self.d['fht'] == 0:  # ask?
                        # *************** 12 month change
                        if self.d['CS'] > 1 and self.d['timelap'] >= 1:
                            self.str += 'k'
                            return Mild
                        elif self.d['CS'] <= 1:  # ask
                            # *************** 12 month change
                            if self.d['discomfort'] > 1 and self.d['timelap'] >= 1:
                                self.str += 'l'
                                return Mild
                            # new change here
                            elif self.d['mobility'] >= 1 and self.d['timelap'] < 1:
                                self.str += 'm'
                                return Mild
                            # none of above
                            # *************** 12 month change
                            if self.d['timelap'] < 1:  # ask:
                                self.str += 'n'
                                return Latent

                else:  # BS>1
                    if self.d['skin'] > 1:
                        if self.d['timelap'] <= 0.5:
                            self.str += 'o'
                            return Mild
                        else:  # timelap>6 months
                            self.str += 'p'
                            return Moderate
                    else:  # skin<=1
                        # below code is what i added
                        # *************** 12 month change
                        if self.d['timelap'] >= 1:
                            self.str += 'q'
                            return Mild
                        elif self.d['mobility'] >= 1 and self.d['timelap'] <= 0.5:  # ask? mobility?
                            self.str += 'r'
                            return Mild
                         # ^^^^ timlap<=6 months condition removed:

                        '''

                        if self.d['mobility'] >=1:# and self.d['timelap'] <=0.5: ##ask? mobility?
                            self.str +='q'
                            return Mild
                        '''
                        '''
                        #code with 12 month fixed to 6 month
                        if self.d['timelap'] >=0.5:
                            self.str +='q'
                            return Mild
                        elif self.d['mobility'] >=1:# and self.d['timelap'] <=0.5: ##ask? mobility?
                            self.str +='r'
                            return Mild
                        '''

            elif self.d['AS'] == 1:
                if self.d['skin'] == 0:
                    self.str += 's'
                    return Mild
                elif self.d['skin'] == 1:
                    self.str += 't'
                    return Moderate
                elif self.d['skin'] > 1 and self.d['fht'] > 0 and self.d['pas'] > 0 and self.d['mobility'] > 0:
                    if 8 <= self.d['symcount'] and self.d['symcount'] <= 12:
                        self.str += 'u'
                        return Moderate
                    elif self.d['symcount'] > 12:
                        self.str += 'v'
                        return Severe

            elif self.d['AS'] > 1:
                if self.d['skin'] > 1 and self.d['fht'] > 0 and self.d['pas'] > 0 and self.d['mobility'] > 0 and self.d['symcount'] > 12:
                    # a bit of change 3.3.2020 1.02 am
                    self.str += 'w'
                    return Severe
                    # if self.d['symcount'] >12:
                    #    self.str +='w'
                    #    return Severe
                else:  # a bit of change
                    self.str += 'x'
                    return Moderate

            #  else:  NONE OF ABOVE
            # still check here
            if self.d['LVC'] > 5:  # CHANGED THIS COLUMN FOR KINECT LDEX VALUES
                # *************** 12 month change
                if self.d['symcount'] > 6 and 0.5 < self.d['timelap']:
                    self.str += 'y'
                    return Mild
                else:
                    self.str += 'z'
                    return Latent
            else:  # Ldex <5
                if self.d['symcount'] > 6 and self.d['fluid'] >= 2:
                    self.str += 'A'
                    return Latent
                else:
                    self.str += 'B'
                    return non_LE

        else:  # LVC >7.1
            # AS ==0
            if self.d['AS'] == 0:
                if self.d['CS'] > 1 and (self.d['fht'] > 0 or self.d['pas'] > 0):
                    self.str += 'C'
                    return Severe
                else:
                    # BS ==0
                    if self.d['BS'] == 0:
                        if self.d['symcount'] < 4:
                            self.str += 'D'
                            return non_LE
                        elif 4 <= self.d['symcount'] and self.d['symcount'] < 6:
                            self.str += 'E'
                            return Latent
                        elif 6 <= self.d['symcount'] and self.d['symcount'] < 10:
                            if self.d['timelap'] <= 0.5:
                                self.str += 'F'
                                return Latent
                            else:
                                self.str += 'G'
                                return Mild
                        else:  # symp_count >=10
                            self.str += 'H'
                            return Moderate
                    # BS >= 1
                    elif self.d['BS'] >= 1:
                        if self.d['mobility'] > 1:
                            self.str += 'I'
                            return Moderate
                        elif self.d['skin'] > 1:
                            self.str += 'J'
                            return Moderate
                        elif self.d['discomfort'] > 1:
                            self.str += 'K'
                            return Moderate
                        # none of Severe above
                        if self.d['BS'] > 1:
                            self.str += 'L'
                            return Moderate
                        if self.d['mobility'] == 1:
                            # *************** 12 month change
                            if self.d['timelap'] <= 0.5:
                                self.str += 'M'
                                return Latent
                            elif self.d['timelap'] > 0.5:
                                self.str += 'N'
                                return Mild

                    if self.d['CS'] > 1 and self.d['fht'] > 0 and self.d['pas'] == 0:
                        # maybe ask this condition?

                        self.str += 'O'
                        return Moderate

            elif self.d['AS'] == 1:
                # is this a division?
                if (self.d['fht'] > 1 or self.d['pas'] > 1) and self.d['BS'] > 0:
                    self.str += 'P'
                    return Severe
                elif self.d['mobility'] > 1:
                    self.str += 'Q'
                    return Severe
                elif self.d['skin'] > 1:
                    self.str += 'R'
                    return Severe
                elif self.d['discomfort'] > 1:
                    self.str += 'S'
                    return Severe
                # if none of above
                if self.d['mobility'] == 1 and self.d['timelap'] > 0.5:
                    self.str += 'T'
                    return Moderate
                if (self.d['fht'] >= 1 or self.d['pas'] >= 1) and self.d['timelap'] > 0.5:
                    self.str += 'U'
                    return Moderate
            elif self.d['AS'] > 1:
                self.str += 'V'
                return Severe
            self.str += 'W'
            # if none above
            return Mild
        self.str += 'X'
        return SPECIAL
        # return non_LE


def get_expert_tree_results(data, class_number=3):

    flags = []
    #data_lst = list(X_arr)
    #row,col = data.shape[0], data.shape[1]
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
        exp_tree = Expert_Tree(row)
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

        if class_number ==3: 
            if expert_tag==1 or expert_tag==0: #none or latent
                result= 0
            elif expert_tag==2: #mild 
                result = 1
            elif expert_tag>2:#moderate severe
                result=2
            else:
                result=-999
        elif class_number ==2: #if not tree clas (if it two-class)
            if expert_tag==1 or expert_tag==0: #none or latent
                result= 0
            elif expert_tag>1: #mild moderate severe
                result = 1
            else:
                result=-999
        elif class_number ==4: #class number 4 
            if expert_tag==1 or expert_tag==0: #none or latent
                result= 0
            elif expert_tag==2: #mild 
                result = 1
            elif expert_tag==3:#moderate 
                result=3
            elif expert_tag==4:#severe 
                result=4
            else:
                result=-999
        else:
            print('invalid class number')
            

        expert_tree_result.append(result)
        flags.append(exp_tree.str)

    return expert_tree_result, expert_truth_labels, flags

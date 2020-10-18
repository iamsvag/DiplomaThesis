# Copyright 2018-2019 CRS4
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class configuration_checker(object):
    def __init__(self, configuration):
        self.configuration = configuration
        self.faults = []

    def check(self):
        self.faults = self.faults + self.check_geometry()
        return len(self.faults) == 0

    def check_geometry(self):
        faults = []
        for d in self.configuration.domains.values():
            n_cells = d['geometry.e_we'] - 1
            p_grid_ratio = d['geometry.parent_grid_ratio']
            if d.parent is not None:
                f = n_cells // p_grid_ratio
                if int(f * p_grid_ratio) != n_cells:
                    faults.append(
                        'Domain %s grid does not fit in its parent grid' %
                        d.name)
        return faults

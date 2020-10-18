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

from .projector import projector


class summarizer(object):
    def __init__(self, configurator):
        self.configurator = configurator
        self.projector = projector(self.configurator['geometry'])
        if self.configurator['dynamics.wrf_core'] == 'ARW':
            self.offset = (
                - ((self.configurator.domains['base']['geometry.e_we'] - 1) *
                   0.5 * self.configurator.domains['base']['geometry.dx']),
                - ((self.configurator.domains['base']['geometry.e_sn'] - 1) *
                   0.5 * self.configurator.domains['base']['geometry.dy']))
        else:
            raise ValueError('Unkown wrf_core %s' %
                             self.configurator['dynamics.wrf_core'])

    def find_lonlat_of_offset(self, p):
        """Find latitude and longitude of point expressed as offset
        from the SW corner of the base domain"""
        return self.projector.project_to_lonlat((p[0] + self.offset[0],
                                                 p[1] + self.offset[1]))

    def summarize(self):
        res = []
        res.append(self.get_global_summary())
        res.append(self.get_domain_descriptions())
        return ''.join(res)

    def get_projection_system_summary(self):
        res = ''
        if self.configurator['geometry.map_proj'] == 'lambert':
            res = """
            Lambert projection
                Reference position ({ref_lon}, {ref_lat})
                True latitudes: [{truelat1}, {truelat2}]
                Standard longitude:  {stand_lon}
            """.format(**self.configurator['geometry'])
        return res

    def get_global_summary(self):
        msg = """
        The map projection system used is:
        {}
        """.format(self.get_projection_system_summary())
        return msg

    def get_domain_descriptions(self):
        res = []
        for dv in self.configurator.domains.values():
            res.append(self.get_domain_description(dv))
        return '\n'.join(res)

    def get_domain_description(self, d):
        offset = d.get_offset_wrt_base()
        delta = d.get_extension()
        sw_corner = self.find_lonlat_of_offset(offset)
        ne_corner = self.find_lonlat_of_offset((offset[0] + delta[0],
                                                offset[1] + delta[1]))
        dpar = "" if d.parent is None else "nested on %s" % d.parent.name
        values = {
            'dname': d.name,
            'dparent': dpar,
            'sw_corner': sw_corner,
            'ne_corner': ne_corner,
            'hcells': d['geometry.e_we'] - 1,
            'vcells': d['geometry.e_sn'] - 1,
            'dx': d['geometry.dx'],
            'dy': d['geometry.dy'],
            'start': d['timespan.start_date'],
            'end': d['timespan.end_date'],
        }
        res = """
        Domain {dname} {dparent}
            South west corner: {sw_corner}
            North east corner: {ne_corner}
            Cells horizontal, dx: {hcells}, {dx}
            Cells vertical,   dy: {vcells}, {dy}
            Will run from {start} to {end}
        """.format(d.name, **values)
        return res

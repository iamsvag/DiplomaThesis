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

# import yaml

from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from .constants import (
    DEFAULTS, SHARE_DEFAULT_FIELDS, GEOGRID_DEFAULT_FIELDS,
    UNGRIB_DEFAULT_FIELDS, METGRID_DEFAULT_FIELDS, GEOMETRY_PROJECTION_FIELDS,
    TIME_CONTROL_DEFAULT_FIELDS, DOMAINS_DEFAULT_FIELDS, FDDA_DEFAULT_FIELDS,
    PHYSICS_DEFAULT_FIELDS, DYNAMICS_DEFAULT_FIELDS, BOUNDARY_CONTROL_FIELDS,
    GRIB2_DEFAULT_FIELDS, NAMELIST_QUILT_DEFAULT_FIELDS)


def merge_configs(base, update):
    assert type(base) == type(update)
    if not isinstance(base, dict):
        return update
    res = {}
    for k in base:
        if k not in update:
            res[k] = base[k]
    for k in update:
        if k not in base:
            res[k] = update[k]
        else:
            res[k] = merge_configs(base[k], update[k])
    return res


class confbox(dict):
    """Holds configuration data as nesteed dictionaries.

    Confobox objects support direct multi level indexing, e.g.,
    ``c['lev1.lev2.lev3'] = v``, both to assign and access values.
    """
    @staticmethod
    def subconfbox(conf):
        for k in conf:
            if isinstance(conf[k], dict):
                conf[k] = confbox(conf[k])

    @staticmethod
    def build_date(cbox):
        now = datetime.utcnow()
        date = datetime(cbox.get('year', now.year),
                        cbox.get('month', now.month),
                        cbox.get('day', now.day),
                        cbox.get('hour', 0),
                        cbox.get('minute', 0),
                        cbox.get('second', 0))
        return date.strftime("%Y-%m-%d_%H:%M:%S")

    def __init__(self, conf):
        super(confbox, self).__init__()
        for k, v in conf.items():
            self[k] = confbox(v) if isinstance(v, dict) else v

    def __setitem__(self, k, v):
        def set_deep(c, p, v):
            if len(p) == 1:
                return super(confbox, c).__setitem__(p[0], v)
            if p[0] not in c:
                c[p[0]] = confbox({})
            return set_deep(c[p[0]], p[1:], v)
        return set_deep(self, k.split('.'), v)

    def __getitem__(self, k):
        def get_deep(c, p):
            if len(p) == 1:
                return c.getattr_base(p[0])
            if p[0] in c:
                return get_deep(c[p[0]], p[1:])
            else:
                raise KeyError(p[0])
        return get_deep(self, k.split('.'))

    def getattr_base(self, k):
        if k in self:
            return super(confbox, self).__getitem__(k)
        if k == 'start_date' and 'start' in self:
            return self.build_date(self['start'])
        if k == 'end_date' and 'end' in self:
            return self.build_date(self['end'])
        if k == 'time_step_seconds' and 'time_step' in self:
            v = Decimal("%s" % self['time_step'])
            return int(v.to_integral_exact())
        if k == 'time_step_fract_num' and 'time_step' in self:
            v = Decimal("%s" % self['time_step'])
            v = Fraction(v - v.to_integral_exact())
            return v.numerator
        if k == 'time_step_fract_den' and 'time_step' in self:
            v = Decimal("%s" % self['time_step'])
            v = Fraction(v - v.to_integral_exact())
            return v.denominator
        raise KeyError(k)


class domain(confbox):
    def __init__(self, name, i, conf):
        super(domain, self).__init__(conf)
        self.parent = None
        self.name = name
        self.id = i

    def __getitem__(self, k):
        try:
            return super(domain, self).__getitem__(k)
        except KeyError as e:
            if k == 'geometry.grid_id':
                return self.id
            # special case:  root domain
            if self.parent is None:
                if k == 'parent_id':
                    return self.id
                elif k in ['running.parent_time_step_ratio',
                           'geometry.parent_grid_ratio',
                           'geometry.i_parent_start',
                           'geometry.j_parent_start']:
                    return 1
                else:
                    raise KeyError(e)
            if k == 'parent_id':
                return self.parent.id
            if k in ['geometry.dx', 'geometry.dy']:
                return self.parent[k] / self['geometry.parent_grid_ratio']
            return self.parent[k]

    def set_parent(self, parent):
        self.parent = parent

    def get_offset_wrt_base(self):
        # WRF uses fortran indices conventions
        ox, oy = 0.0, 0.0
        d = self
        while d.parent is not None:
            i_offset = d['geometry.i_parent_start'] - 1
            j_offset = d['geometry.j_parent_start'] - 1
            ox = ox + i_offset * d.parent['geometry.dx']
            oy = oy + j_offset * d.parent['geometry.dy']
            d = d.parent
        return (ox, oy)

    def get_extension(self):
        return ((self['geometry.e_we'] - 1) * self['geometry.dx'],
                (self['geometry.e_sn'] - 1) * self['geometry.dy'])


def split_key(dk):
    if dk.startswith('@'):
        return tuple(dk[1:].split('.', 1))
    else:
        return None, dk


class configurator(confbox):

    @staticmethod
    def gather_domains_info(conf):
        domains = {}
        for i, (dn, v) in enumerate(conf['domains'].items()):
            domains[dn] = domain(dn, i + 1, v)

        for v in domains.values():
            if 'parent' in v:
                v.set_parent(domains[v['parent']])
                del(v['parent'])
        return domains

    @classmethod
    def make(cls, assigned=None):
        return cls(merge_configs(
            DEFAULTS, {} if assigned is None else assigned))

    def __init__(self, conf):
        super(configurator, self).__init__(conf['global'])
        self.domains = self.gather_domains_info(conf)
        self.domains_sequence = list(
            map(lambda x: x[0], sorted([(n, self.domains[n].id)
                                        for n in self.domains],
                                       key=lambda x: x[1])))

    def __getitem__(self, k):
        dn, k = split_key(k)
        if dn is None:
            return super(configurator, self).__getitem__(k)
        else:
            return self.domains[dn][k]

    def __setitem__(self, k, v):
        dn, k = split_key(k)
        if dn is None:
            return super(configurator, self).__setitem__(k, v)
        else:
            if dn not in self.domains:
                self.domains_sequence.append(dn)
                self.domains[dn] = domain(dn, len(self.domains), {})
                self.domains[dn].set_parent(self.domains['base'])
            self.domains[dn][k] = v

    def update(self, argkv):
        for k, v in argkv.items():
            self[k] = v

    def gather_data(self, tags, ignore_if_missing=True):
        def normalize(t):
            if isinstance(t, tuple):
                return t
            else:
                return (t, t.split('.')[-1])

        def helper(t):
            t, k = t
            if t == 'domains.max_dom':
                return ('max_dom', len(self.domains))
            if t == 'domains.geometry.boundary.specified':
                return ('specified',
                        [n == 'base' for n in self.domains_sequence])
            if t == 'domains.geometry.boundary.nested':
                return ('nested', [n != 'base' for n in self.domains_sequence])
            p = t.split('.')
            if p[0] == 'domains':
                t = '.'.join(p[1:])
                v = [self.domains[n][t]
                     for n in self.domains_sequence]
            elif p[0] in self.domains:
                t = '.'.join(p[1:])
                v = self.domains[p[0]][t]
            else:
                v = self[t]
            return (k, v)

        def wrap_helper(t):
            try:
                return helper(t)
            except KeyError as e:
                if ignore_if_missing:
                    return (None, None)
                else:
                    raise e

        return dict(_ for _ in (wrap_helper(normalize(_)) for _ in tags)
                    if _[0] is not None)

    def generate_section(self, sname, fields):

        def format_value(v):
            if isinstance(v, list):
                return ', '.join(map(format_value, v))
            elif isinstance(v, bool):
                return '.true.' if v else '.false.'
            elif isinstance(v, (int, float)):
                return '{}'.format(v)
            else:
                return "'{}'".format(v)
        body = ',\n '.join('{} = {}'.format(fn, format_value(fields[fn]))
                           for fn in fields)
        return '&{}\n {}\n/\n'.format(sname, body)

    def generate_share(self):
        fields = self.gather_data(SHARE_DEFAULT_FIELDS)
        return self.generate_section('share', fields)

    def generate_geogrid(self):
        fields = self.gather_data(GEOGRID_DEFAULT_FIELDS)
        projection = self['geometry.map_proj']
        fields.update([('map_proj', self['geometry.map_proj'])])
        fields.update(self.gather_data(
            GEOMETRY_PROJECTION_FIELDS[projection]))
        return self.generate_section('geogrid', fields)

    def generate_ungrib(self):
        fields = self.gather_data(UNGRIB_DEFAULT_FIELDS)
        return self.generate_section('ungrib', fields)

    def generate_metgrid(self):
        fields = self.gather_data(METGRID_DEFAULT_FIELDS)
        return self.generate_section('metgrid', fields)

    def generate_fdda(self):
        fields = self.gather_data(FDDA_DEFAULT_FIELDS)
        return self.generate_section('fdda', fields)

    def generate_domains(self):
        fields = self.gather_data(DOMAINS_DEFAULT_FIELDS)
        return self.generate_section('domains', fields)

    def generate_physics(self):
        fields = self.gather_data(PHYSICS_DEFAULT_FIELDS)
        return self.generate_section('physics', fields)

    def generate_time_control(self):
        fields = self.gather_data(TIME_CONTROL_DEFAULT_FIELDS)
        return self.generate_section('time_control', fields)

    def generate_dynamics(self):
        fields = self.gather_data(DYNAMICS_DEFAULT_FIELDS)
        return self.generate_section('dynamics', fields)

    def generate_bdy_control(self):
        fields = self.gather_data(BOUNDARY_CONTROL_FIELDS)
        return self.generate_section('bdy_control', fields)

    def generate_grib2(self):
        fields = self.gather_data(GRIB2_DEFAULT_FIELDS)
        return self.generate_section('grib2', fields)

    def generate_namelist_quilt(self):
        fields = self.gather_data(NAMELIST_QUILT_DEFAULT_FIELDS)
        return self.generate_section('namelist_quilt', fields)

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

SHARE_DEFAULT_FIELDS = [
    'dynamics.wrf_core', 'domains.max_dom',
    'domains.timespan.start_date',
    'domains.timespan.end_date',
    'running.input.interval_seconds',
    ('geogrid.io_form', 'io_form_geogrid'),
]

GEOGRID_DEFAULT_FIELDS = [
    'domains.parent_id',
    'domains.geometry.parent_grid_ratio',
    'domains.geometry.i_parent_start',
    'domains.geometry.j_parent_start',
    'domains.geometry.e_we',
    'domains.geometry.e_sn',
    'domains.geometry.geog_data_res',
    '@base.geometry.dx',
    '@base.geometry.dy',
    'geometry.geog_data_path',
    'geometry.opt_geogrid_tbl_path',
]

GEOMETRY_PROJECTION_FIELDS = {
    'lambert': [
        'geometry.ref_lat',
        'geometry.ref_lon',
        'geometry.truelat1',
        'geometry.truelat2',
        'geometry.stand_lon'
    ]
}


UNGRIB_DEFAULT_FIELDS = [
    'ungrib.out_format',
    'ungrib.prefix',
]

METGRID_DEFAULT_FIELDS = [
    'metgrid.fg_name',
    'metgrid.io_form_metgrid',
    'metgrid.opt_metgrid_tbl_path',
]

FDDA_DEFAULT_FIELDS = [
]

TIME_CONTROL_DEFAULT_FIELDS = [
    ('domains.timespan.start.year', 'start_year'),
    ('domains.timespan.start.month', 'start_month'),
    ('domains.timespan.start.day', 'start_day'),
    ('domains.timespan.start.hour', 'start_hour'),
    ('domains.timespan.start.minute', 'start_minute'),
    ('domains.timespan.start.second', 'start_second'),
    ('domains.timespan.end.year', 'end_year'),
    ('domains.timespan.end.month', 'end_month'),
    ('domains.timespan.end.day', 'end_day'),
    ('domains.timespan.end.hour', 'end_hour'),
    ('domains.timespan.end.minute', 'end_minute'),
    ('domains.timespan.end.second', 'end_second'),
    'running.input.interval_seconds',
    ('domains.running.history.interval', 'history_interval'),
    ('running.history.io_form', 'io_form_history'),
    'domains.running.output.frames_per_outfile',
    'domains.running.input.input_from_file',
    'running.input.restart',
    'running.input.io_form_restart',
    'running.input.io_form_input',
    'running.input.io_form_boundary',
    ('running.debug.level', 'debug_level'),
    ('running.history.outname', 'history_outname'),
]

DOMAINS_DEFAULT_FIELDS = [
    ('running.time_step_seconds', 'time_step'),
    'running.time_step_fract_num',
    'running.time_step_fract_den',
    'running.feedback',
    'running.smooth_option',
    'domains.max_dom',
    'domains.parent_id',
    'domains.geometry.grid_id',
    'domains.geometry.e_vert',
    'domains.geometry.e_we',
    'domains.geometry.e_sn',
    'domains.geometry.dx',
    'domains.geometry.dy',
    'domains.geometry.i_parent_start',
    'domains.geometry.j_parent_start',
    'domains.geometry.parent_grid_ratio',
    'domains.running.parent_time_step_ratio',
    'real.num_metgrid_levels',
    'real.num_metgrid_soil_levels',
    'real.eta_levels',
    'running.parallel.numtiles',
]

PHYSICS_DEFAULT_FIELDS = [
    ('domains.physics.mp', 'mp_physics'),
    ('domains.physics.ra_lw', 'ra_lw_physics'),
    ('domains.physics.ra_sw', 'ra_sw_physics'),
    'domains.physics.radt',
    ('domains.physics.sf_sfclay', 'sf_sfclay_physics'),
    ('domains.physics.sf_surface', 'sf_surface_physics'),
    ('domains.physics.bl_pbl', 'bl_pbl_physics'),
    ('domains.physics.cu', 'cu_physics'),
    'domains.physics.cudt',
    'physics.num_soil_layers',
    'physics.num_land_cat',
    'physics.surface_input_source',
]

DYNAMICS_DEFAULT_FIELDS = [
    'dynamics.rk_ord',
    'domains.dynamics.diff_opt',
    'domains.dynamics.km_opt',
    'domains.dynamics.non_hydrostatic',
]

BOUNDARY_CONTROL_FIELDS = [
    'geometry.boundary.spec_bdy_width',
    'geometry.boundary.spec_zone',
    'geometry.boundary.relax_zone',
    'domains.geometry.boundary.specified',
    'domains.geometry.boundary.nested',
]


GRIB2_DEFAULT_FIELDS = [
]

NAMELIST_QUILT_DEFAULT_FIELDS = [
    'running.parallel.nio_tasks_per_group',
    'running.parallel.nio_groups'
]

DEFAULTS = {
    'global': {
        'geometry': {
            'map_proj': 'lambert',
            'opt_geogrid_tbl_path': '/wrf/WPS/geogrid',
            'boundary': {
                'spec_bdy_width': 5,
                'spec_zone': 1,
                'relax_zone': 4,
                'constant_bc': False,
                'spec_exp': 0.33,
            },
        },
        'dynamics': {
            'wrf_core': 'ARW',
            'rk_ord': 3,
        },
        'running': {
            'debug': {
                'level': 0,
            },
            'input': {
                'restart': False,
                'io_form_input': 2,
                'io_form_restart': 2,
                'io_form_boundary': 2,
            },
            'history': {
                'outname': "/run/wrfout_d<domain>_<date>.nc",
                'io_form': 2,
            },
            'feedback': 0,
            'smooth_option': 2,
            'parallel': {
                'nio_tasks_per_group': 0,
                'nio_groups': 1,
                'numtiles': 1,
            },
        },
        'geogrid': {
            'io_form': 2,
            'opt_output_from_geogrid_path': '.',
        },
        'ungrib': {
            'out_format': 'WPS',
            'prefix': 'FILE',
        },
        'metgrid': {
            'fg_name': 'FILE',
            'io_form_metgrid': 2,
            'opt_metgrid_tbl_path': '/wrf/WPS/metgrid',
        },
        'physics': {
            'swint_opt': 0,
            'convtrans_avglen_m': 20,
            'ishallow': 0,
            'surface_input_source': 1,
            'num_soil_layers': 5,
            'num_land_cat': 21,
            'maxiens': 1,
            'maxens': 3,
            'maxens2': 3,
            'maxens3': 16,
            'ensdim': 144,
            'mp_zero_out': 0,
            'usemonalb': False,
            'mosaic_lu': 0,
            'mosaic_soil': 0,
        },
    },
    'domains': {
        'base': {
            'timespan': {
                'start': {
                    'year': 0,
                    'month': 0,
                    'day': 0,
                    'hour': 0,
                    'minute': 0,
                    'second': 0,
                },
                'end': {
                    'year': 0,
                    'month': 0,
                    'day': 0,
                    'hour': 0,
                    'minute': 0,
                    'second': 0,
                },
            },
            'running': {
                'output': {
                    'frames_per_outfile': 1,
                },
                'input': {
                    'input_from_file': True,
                },
            },
            'physics': {
                'mp': 0,
                'ra_lw': 0,
                'ra_sw': 0,
                'sf_sfclay': 0,
                'sf_surface': 0,
                'bl_pbl': 0,
                'bldt': 0,
                'cu': 0,
                'cu_diag': 0,
                'cudt': 0,
                'radt': 0,
                'cu_rad_feedback': False,
            },
            'dynamics': {
                'diff_opt': 1,
                'km_opt': 1,
                'non_hydrostatic': True,
            },
        },
    },
}

import numpy as np
from datetime import datetime
from scipy.spatial import KDTree
import xmltodict
# No instalar shapely desde los repositorios de pip, usar wheel
# https://towardsdatascience.com/install-shapely-on-windows-72b6581bb46c
import shapely.geometry
import re
from crdp import rdp
import geojson
from geojson import Feature, LineString, MultiLineString, Polygon, Point
from turfpy import measurement
from typing import List, Union
import matplotlib.pyplot as plt


class Duct:
    R = 6378137.0  # WGS84 Equatorial Radius in Meters

    def __init__(self, filename: str()):
        with open(filename, encoding='utf8') as fileData:
            self.duct = geojson.load(fileData)
            self.name = self.duct.properties['name']
            self.linestring = self.duct.geometry
            self.length = measurement.length(self.linestring, 'm')
            self.coordinates = self.linestring.coordinates

    @property
    def tree(self) -> KDTree:
        if not hasattr(self, '_tree'):
            self._tree = KDTree(self.coordinates)
        return self._tree

    @property
    def covered(self) -> Union[Feature, None]:
        return self._covered

    @covered.setter
    def covered(self, value):
        if value == None:
            self._covered = value
            self.covered_pctg = 0
            self.not_covered = Feature(
                geometry=shapely.geometry.MultiLineString([self.coordinates]))
        else:
            self._covered = value
            self.covered_pctg = 100 * measurement.length(
                self.covered, 'm') / self.length
            self.not_covered = self.set_not_covered()

    @staticmethod
    def _line_equation(p1: tuple, p2: tuple) -> tuple:
        # Ecuación de línea y=a*x+b
        a = (p1[1] - p2[1]) / (p1[0] - p2[0])
        b = p1[1] - a * p1[0]
        return a, b

    @staticmethod
    def intersects(line, poly):
        test = (poly[0][1] * line[0] + line[1]) > poly[0][0]
        for p in poly[1:]:
            if test != ((p[1] * line[0] + line[1]) > p[0]):
                return True
        return False

    class File:
        @staticmethod
        def get_tags(file, end='little'):
            # https://www.loc.gov/preservation/digital/formats/content/tiff_tags.shtml

            # Offset al primer IFD
            file.seek(4)
            ifd = int.from_bytes(file.read(4), end)

            # Número de tags
            file.seek(ifd)
            entries = int.from_bytes(file.read(2), end)

            tags = []
            for i in range(entries):
                off = ifd + 2 + 12*i
                file.seek(off)
                tag = int.from_bytes(file.read(2), end)
                typ = int.from_bytes(file.read(2), end)
                cnt = int.from_bytes(file.read(4), end)
                val = int.from_bytes(file.read(4), end)
                tags.append({
                    'tag': tag,
                    'type': typ,
                    'count': cnt,
                    'value': val
                })
            return tags

        @staticmethod
        def read_tiff(file, row, col, samples):
            # https://docs.fileformat.com/image/tiff/
            # Endianness
            end = 'big' if file.read(2) == b'MM' else 'little'
            index = row * samples + col

            tags = Duct.File.get_tags(file, end)
            pointer = next((t for t in tags if t['tag'] == 273), None)['value']

            # Inicio de data
            file.seek(pointer)
            start = int.from_bytes(file.read(4), end)

            # Búsqueda del valor
            file.seek(start + 2*index)
            val = int.from_bytes(file.read(2), end, signed=True)
            return val

        @staticmethod
        def AW3D30_name(point):
            lat = ('N' if point.coordinates[1] > 0 else 'S') + \
                '{:03d}'.format(int(abs(point.coordinates[1]//1)))
            lon = ('E' if point.coordinates[0] > 0 else 'W') + \
                '{:03d}'.format(int(abs(point.coordinates[0]//1)))
            return f'ALPSMLC30_{lat}{lon}_DSM.tif'

    @staticmethod
    def elevation(point: Point, samples: float = 3600):
        filename = 'AW3D30/' + Duct.File.AW3D30_name(point)
        with open(filename, 'rb') as file:
            row = samples - 1 - int((point.coordinates[1] % 1) * samples)
            col = int((point.coordinates[0] % 1) * samples)
            elev = Duct.File.read_tiff(file, row, col, samples)
            return elev

    def distance_to_line(self, point: Point) -> float:
        point = Feature(geometry=point)
        return measurement.point_to_line_distance(point, self.linestring, 'm')

    def along(self, distance: float, direction: int = 1) -> Point:
        assert direction in [1, -1], '"direction" debería ser 1 o -1'
        linestring = LineString(
            self.coordinates[::direction])
        return measurement.along(linestring, distance, 'm')

    def route_between(self,
                      p1: Union[Point, List[float]],
                      p2: Union[Point, List[float]],
                      addAltitude: bool = True,
                      altitude: float = 120,
                      epsilon: float = 0,
                      absolute: bool = False,
                      baseAltitude: float = 0,
                      ) -> LineString:
        if type(p1) == 'geojson.geometry.Point':
            p1 = p1.coordinates
        if type(p2) == 'geojson.geometry.Point':
            p2 = p2.coordinates

        idx1 = self.tree.query(p1)[1]
        idx2 = self.tree.query(p2)[1]

        if idx1 < idx2:
            if idx2 != len(self.coordinates):
                points = [p1] + self.coordinates[idx1+1:idx2] + [p2]
            else:
                points = [p1] + self.coordinates[idx1+1:] + [p2]
        else:
            if idx2 != 0:
                points = [p1] + self.coordinates[idx1-1:idx2:-1] + [p2]
            else:
                points = [p1] + self.coordinates[idx1-1::-1] + [p2]

        if addAltitude:
            if absolute:
                points = [(p[0], p[1], p[2]+altitude-baseAltitude)
                          for p in points]
                for point in points:
                    if point[2] >= 500:
                        print(
                            f'Advertencia, el dron se elevará {point[2]} m desde el punto de despegue.')
            else:
                points = [(p[0], p[1], altitude-baseAltitude) for p in points]

        points_rdp = rdp(points, epsilon=epsilon)
        return LineString(points_rdp)

    def route_distance(self,
                       start: Union[Point, List[float]],
                       distance: float,
                       altitude: float = 120,
                       epsilon: float = 0,
                       absolute: bool = False,
                       baseAltitude: float = 0,
                       inverse: bool = False,
                       ) -> LineString:
        if type(start) == 'geojson.geometry.Point':
            start = start.coordinates
        idx = self.tree.query(start)[1]
        if inverse:
            idx = max(idx-1, 0)
            ls = LineString(self.coordinates[-1:idx:-1])
        else:
            ls = LineString(self.coordinates[idx:])
        end = measurement.along(ls, distance, 'm').geometry
        return self.route_between(start, end, altitude, epsilon, absolute, baseAltitude)

    @staticmethod
    def generate_KML(route, name, absolute: bool() = False):
        routeStr = ' '.join(['{},{},{}'.format(*c) for c in route.coordinates])
        xml = {
            'Document': {
                'name': name,
                'Placemark': [
                    {
                        'name': name,
                        'Style': {'IconStyle': {'color': 'ff0000ff'}},
                        'Point': {
                            'altitudeMode': 'absolute' if absolute else 'relativeToGround',
                            'coordinates': '{},{},{}'.format(*route.coordinates[0])
                        }
                    },
                    {
                        'name': name,
                        'Style': {'LineStyle': {'color': 'ff00ffff', 'width': 5}},
                        'LineString': {
                            'altitudeMode': 'absolute' if absolute else 'relativeToGround',
                            'coordinates': routeStr
                        }
                    }
                ]
            }
        }
        with open(name+'.kml', 'w') as f:
            XML = xmltodict.unparse(xml, pretty=True)
            f.write(XML)

    def coverage(self, photos: shapely.geometry.multipolygon.MultiPolygon):
        result = photos.intersection(
            shapely.geometry.LineString(self.coordinates))
        if result.type == 'GeometryCollection':
            return None
        return Feature(geometry=shapely.geometry.mapping(result))

    def set_not_covered(self, properties: dict = {}) -> Feature:
        if self.covered != None:
            assert self.covered.geometry.type in [
                'LineString', 'MultiLineString'], \
                '"geometry.type" debería ser "LineString" o "MultiLineString'

        missing = []
        if self.covered.geometry.type == 'LineString':
            coords = [self.covered.geometry.coordinates]
        else:
            coords = self.covered.geometry.coordinates

        if coords[0][0] != self.coordinates[0]:
            # El inicio del ducto no está cubierto
            line = self.route_between(
                self.coordinates[0], coords[0][0], addAltitude=False)
            if len(line.coordinates) > 1:
                missing.append(line.coordinates)

        for i in range(len(coords)-1):
            line = self.route_between(
                coords[i][-1], coords[i+1][0], addAltitude=False)
            if len(line.coordinates) > 1:
                missing.append(line.coordinates)

        if coords[-1][-1] != self.coordinates[-1]:
            # El final del ducto no está cubierto
            line = self.route_between(
                coords[-1][-1], self.coordinates[-1], addAltitude=False)
            if len(line.coordinates) > 1:
                missing.append(line.coordinates)

        geometry = MultiLineString(coordinates=missing)
        return Feature(geometry=geometry, properties=properties)


class Photo:
    def __init__(self, **kwargs):
        if 'heading' in kwargs:
            kwargs['theta'] = np.deg2rad(kwargs['heading'])
        if 'datetime' in kwargs:
            kwargs['timestamp'] = kwargs['datetime']
        self.latitude = kwargs['latitude']
        self.longitude = kwargs['longitude']
        self.altitude = kwargs['altitude']
        self.gsd = kwargs['gsd']
        self.theta = kwargs['theta']
        self.width = kwargs['width']
        self.height = kwargs['height']
        if 'url' in kwargs.keys():
            self.url = kwargs['url']
        if kwargs['timestamp']:
            m = re.match(r'(.*)\.000', kwargs['timestamp'])
            if m:
                kwargs['timestamp'] = m.group(1)
            self.timestamp = datetime.strptime(
                kwargs['timestamp'], '%Y-%m-%d %H:%M:%S')
        self.center = Point((self.longitude, self.latitude))
        self.corners = self.set_corners()

    def set_corners(self) -> List[Point]:
        w = self.width/2
        h = self.height/2
        p1 = self.map_point(w, h)
        p2 = self.map_point(w, -h)
        p3 = self.map_point(-w, -h)
        p4 = self.map_point(-w, h)
        return [p1, p2, p3, p4]

    @property
    def ground_height(self):
        if not hasattr(self, '_ground_height'):
            self._ground_height = round(
                self.altitude - Duct.elevation(self.center), 2)
        return self._ground_height

    @property
    def size(self):
        return self.width, self.height

    @property
    def polygon(self):
        if not hasattr(self, '_polygon'):
            self._polygon = shapely.geometry.Polygon(self.corners)
            # self._polygon = shapely.geometry.Polygon(
            #     self.corners + [self.corners[0]])
        return self._polygon

    @property
    def geo_polygon(self):
        if not hasattr(self, '_geo_polygon'):
            polygon = [self.corners + self.corners[0:1]]
            self._geo_polygon = Feature(geometry=Polygon(polygon))
        return self._geo_polygon

    def scale(self, x: int(), y: int()) -> tuple():
        return self.gsd*x, self.gsd*y

    def rotate(self, point: tuple(), angle: float, origin: tuple() = (0, 0)) -> tuple():
        ox, oy = origin
        px, py = point
        qx = ox + np.cos(angle) * (px - ox) + np.sin(angle) * (py - oy)
        qy = oy - np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy

    def move(self, point: tuple(), vector: tuple()) -> tuple():
        R = 6378137.0   # WGS84 Equatorial Radius in Meters
        lat = point[1] + np.rad2deg(vector[1] / R)
        lon = point[0] + np.rad2deg(vector[0] / R) / np.cos(np.deg2rad(lat))
        return lon, lat

    def map_point(self, x: int(), y: int()) -> tuple():
        scaled = self.scale(x, y)
        rotated = self.rotate(scaled, self.theta)
        moved = self.move(self.center.coordinates, rotated)
        return moved

    def draw(self) -> dict:
        coordinates = ' '.join([f'{p[0]},{p[1]},0' for p in self.corners])
        return {
            'name': str(self.timestamp),
            'Polygon': {
                'outerBoundaryIs': {
                    'LinearRing': {
                        'coordinates': coordinates,
                        'altitudeMode': 'relativeToGround'
                    }
                }
            },
            'styleUrl': '#photoStyle'
        }


class Flight:
    def __init__(self, data=[], photos=[]):
        self.data = data
        self.photos = sorted(photos, key=lambda x: x.timestamp)

    def draw_not_covered_kml(self):
        lines = []
        for line in self.data:
            lines += line['duct'].not_covered.geometry.coordinates

        segments = []
        for line in lines:
            segments.append({
                'Point': {
                    'coordinates': '{},{},0'.format(*line[0]),
                    'altitudeMode': 'relativeToGround'
                },
                'styleUrl': '#missingStyle'
            })
            segments.append({
                'LineString': {
                    'coordinates': ' '.join([f'{p[0]},{p[1]},0' for p in line]),
                    'altitudeMode': 'relativeToGround'
                },
                'styleUrl': '#missingStyle'
            })
            segments.append({
                'Point': {
                    'coordinates': '{},{},0'.format(*line[-1]),
                    'altitudeMode': 'relativeToGround'
                },
                'styleUrl': '#missingStyle'
            })

        return segments

    def draw_not_covered(self):
        lines = []

        # Descompone multilinestrings en una lista de linestrings
        # de todos los ductos
        for line in self.data:
            lines += line['duct'].not_covered.geometry.coordinates

        # Para cada segmento no cubierto, gener un punto al inicio
        # y otro al final
        segments = []
        for line in lines:
            segments.append(Feature(geometry=Point(line[0])))
            segments.append(Feature(geometry=LineString(line)))
            segments.append(Feature(geometry=Point(line[-1])))

        return segments

    def draw_photos(self):
        features = []
        for photo in self.photos:
            features.append(photo.geo_polygon)
        return features

    def kml(self, filename):
        if self.photos[0].timestamp.date() == self.photos[-1].timestamp.date():
            name = f'Cobertura del {self.photos[0].timestamp.date()}'
        else:
            name = f'Cobertura entre {self.photos[0].timestamp.date()} y {self.photos[-1].timestamp.date()}'
        xml = {
            'Document': {
                'name': name,
                'Style': [
                    {
                        '@id': 'photoStyle',
                        'LineStyle': {'color': 'ff00ffff'},
                        'PolyStyle': {'color': '8000ffff'}
                    },
                    {
                        '@id': 'missingStyle',
                        'IconStyle': {'color': 'ff0000ff'},
                        'LineStyle': {'color': 'ff0000ff', 'width': 5}
                    },
                ],
                'Folder': [
                    {
                        'name': 'Fotos',
                        'open': 0,
                        'Placemark': [photo.draw() for photo in self.photos]
                    },
                    {
                        'name': 'Puntos no cubiertos',
                        'open': 0,
                        'Placemark': self.draw_not_covered_kml()
                    },
                ]
            }
        }
        with open(filename, 'w') as f:
            XML = xmltodict.unparse(xml, pretty=True)
            f.write(XML)

    def get_times(self):
        if not any(['photos' in place.keys() for place in self.data]):
            return
        print('Hora:')
        for place in self.data:
            if 'photos' not in place.keys() or len(place['photos']) == 0:
                continue
            name = place['duct'].name
            photos = sorted(place['photos'], key=lambda x: x.timestamp)
            times = [photos[0].timestamp, photos[-1].timestamp]
            print('\tInicio {}\t: {}'.format(name, times[0]))
            print('\tTérmino {}\t: {}'.format(name, times[-1]))

    def get_altitudes(self, min_altitude=10):
        alts = [f.altitude for f in self.photos if f.altitude > min_altitude]
        print('Altitud:')
        print('\tMáxima\t: {:4d} msnm'.format(round(max(alts))))
        print('\tMínima\t: {:4d} msnm'.format(round(min(alts))))
        print('\tMedia\t: {:4d} msnm'.format(round(sum(alts)/len(alts))))

    def get_coverage(self):
        print('Porcentaje de cobertura:')
        total_covered = 0
        total_length = 0

        for place in [p for p in self.data if 'optional' not in p.keys() or p['optional'] == False]:
            duct = place['duct']
            total_covered += duct.length * duct.covered_pctg / 100
            total_length += duct.length
            print(f"\t{duct.name}\t: {duct.covered_pctg:.2f}%")

        print('\tTotal\t: {:.2f}%'.format(100*total_covered/total_length))

        for place in [p for p in self.data if 'optional' in p.keys() and p['optional'] == True]:
            duct = place['duct']
            total_covered += duct.length * duct.covered_pctg / 100
            total_length += duct.length
            print(f"\t{duct.name}\t: {duct.covered_pctg:.2f}%")

    def get_stats(self):
        self.get_times()
        self.get_altitudes()
        self.get_coverage()
        # self.get_parallels()
        if any(['photos' in place.keys() for place in self.data]):
            print('Total de fotos:')
            for place in self.data:
                if 'photos' not in place.keys() or len(place['photos']) == 0:
                    continue
                print('\t{}\t: {}'.format(
                    place['duct'].name, len(place['photos'])))

    def make_line_chart(self, ax):
        for place in self.data:
            heights = [photo.ground_height for photo in place['photos']]
            ax.plot(range(len(heights)), heights, label=place['duct'].name)
        ax.set_title('Perfil de altura')
        ax.set_xlabel('# de foto')
        ax.set_ylabel('Altura [m]')
        ax.legend()
        ax.grid()

    def make_box_plot(self, ax):
        tmp = {place['duct'].name: [ph.ground_height for ph in place['photos']]
               for place in self.data if 'optional' not in place.keys() or place['optional'] == False}
        ax.boxplot(list(tmp.values()), labels=list(tmp.keys()))
        ax.set_title('Distribución de alturas')
        ax.set_ylabel('Altura [m]')
        ax.grid()

    def make_hist(self, ax):
        heights = [p.ground_height for p in self.photos]
        bins = [i for i in range(int(min(heights)), int(max(heights))+1, 100)]
        ax.hist(heights, bins=bins)
        ax.set_title('Distribución de alturas')
        ax.set_xlabel('Altura [m]')
        ax.set_ylabel('Cantidad de imágenes')
        ax.grid()

    def plot(self, date=None, figsize: tuple = (16, 6)):
        fig = plt.figure(constrained_layout=True, figsize=figsize)
        subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[2.5, 1])
        if date != None:
            fig.suptitle(f'Perfil de altura vuelo {date}', fontsize=16)
        else:
            fig.suptitle('Perfil de altura', fontsize=16)

        ax11 = subfigs[0].subplots(1, 1)
        (ax21, ax22) = subfigs[1].subplots(2, 1)

        self.make_line_chart(ax11)
        self.make_box_plot(ax21)
        self.make_hist(ax22)

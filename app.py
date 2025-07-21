import streamlit as st
import geopandas as gpd
import leafmap.foliumap as leafmap
import rasterio
from rasterio.plot import show
import numpy as np
import os
from tempfile import NamedTemporaryFile
import ezdxf
from shapely.geometry import LineString
import pydeck as pdk
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
from rasterio import Affine
from rasterio.transform import from_origin

st.set_page_config(layout="wide")
st.title("🗺️ Geospatial Viewer: Shapefiles, DXF, DEMs, Terrain & Analysis")

st.sidebar.header("Upload Geospatial Data")

# File uploaders
uploaded_shp = st.sidebar.file_uploader("Upload SHP/DBF files (ZIP recommended)", type=["zip"])
uploaded_dxf = st.sidebar.file_uploader("Upload DXF file", type=["dxf"])
uploaded_dem = st.sidebar.file_uploader("Upload DEM GeoTIFF", type=["tif"])
uploaded_slope = st.sidebar.file_uploader("Upload Precomputed Slope Raster (optional)", type=["tif"])

show_3d = st.sidebar.checkbox("Enable 3D Terrain View (pydeck)", value=False)
show_layers = {
    "shapefile": st.sidebar.checkbox("Show Shapefile Layer", value=True),
    "dxf": st.sidebar.checkbox("Show DXF Layers", value=True),
    "dem": st.sidebar.checkbox("Show DEM Raster", value=True),
    "slope": st.sidebar.checkbox("Show Precomputed Slope Raster", value=False),
}

m = leafmap.Map(center=[1.25, 103.83], zoom=14)

# For PDF export
pdf_images = []

# Process shapefile zip
if uploaded_shp is not None:
    with NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        tmp_zip.write(uploaded_shp.read())
        tmp_zip_path = tmp_zip.name

    gdf = gpd.read_file(f"zip://{tmp_zip_path}")
    opacity = st.sidebar.slider("SHP Layer Opacity", 0.0, 1.0, 0.8)
    if show_layers["shapefile"]:
        m.add_gdf(gdf, layer_name="Uploaded SHP", opacity=opacity)
    st.sidebar.markdown("**SHP Metadata**")
    st.sidebar.json(gdf.dtypes.astype(str).to_dict())

    geojson_buf = BytesIO()
    gdf.to_file(geojson_buf, driver="GeoJSON")
    geojson_buf.seek(0)
    st.download_button("Download Shapefile as GeoJSON", geojson_buf, file_name="shapefile.geojson", mime="application/geo+json")

# Process DXF with layer support
if uploaded_dxf is not None:
    with NamedTemporaryFile(delete=False, suffix=".dxf") as tmp_dxf:
        tmp_dxf.write(uploaded_dxf.read())
        tmp_dxf_path = tmp_dxf.name

    try:
        doc = ezdxf.readfile(tmp_dxf_path)
        msp = doc.modelspace()
        layers = {}
        for e in msp:
            if e.dxftype() == 'LINE':
                layer_name = e.dxf.layer
                line = LineString([e.dxf.start, e.dxf.end])
                if layer_name not in layers:
                    layers[layer_name] = []
                layers[layer_name].append(line)

        for layer, lines in layers.items():
            dxf_gdf = gpd.GeoDataFrame(geometry=lines)
            dxf_gdf.set_crs(epsg=4326, inplace=True, allow_override=True)
            layer_opacity = st.sidebar.slider(f"{layer} Opacity", 0.0, 1.0, 0.8)
            if show_layers["dxf"]:
                m.add_gdf(dxf_gdf, layer_name=f"DXF - {layer}", opacity=layer_opacity)
    except Exception as e:
        st.warning(f"DXF read error: {e}")

# Show optional precomputed slope raster
if uploaded_slope is not None and show_layers["slope"]:
    with NamedTemporaryFile(delete=False, suffix=".tif") as tmp_slope:
        tmp_slope.write(uploaded_slope.read())
        tmp_slope_path = tmp_slope.name

    try:
        with rasterio.open(tmp_slope_path) as slope_src:
            slope_data = slope_src.read(1)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_title("Precomputed Slope (degrees)")
            img = ax.imshow(slope_data, cmap="viridis")
            plt.colorbar(img, ax=ax, shrink=0.6)
            ax.axis('off')
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"Slope raster error: {e}")

# Process DEM raster and derived products
if uploaded_dem is not None:
    with NamedTemporaryFile(delete=False, suffix=".tif") as tmp_tif:
        tmp_tif.write(uploaded_dem.read())
        tmp_tif_path = tmp_tif.name

    try:
        if show_layers["dem"]:
            try:
                import localtileserver
                m.add_raster(tmp_tif_path, layer_name="DEM Raster", colormap="terrain", opacity=0.6, port=0)
            except ImportError:
                st.info("🔍 DEM raster display skipped: 'localtileserver' is not installed. Install it locally to enable tiled DEM viewing.")
            except Exception as e:
                st.warning(f"DEM display error: {e}")
        except ImportError:
            st.info("🔍 DEM raster display skipped: 'localtileserver' is not installed. Install it locally to enable tiled DEM viewing.")
        except Exception as e:
            st.warning(f"DEM display error: {e}")

        with rasterio.open(tmp_tif_path) as src:
            profile = src.profile
            elevation = src.read(1)
            transform = src.transform
            x, y = np.gradient(elevation)
            slope = np.pi/2. - np.arctan(np.hypot(x, y))
            aspect = np.arctan2(-x, y)
            azimuth = 315.0 * np.pi / 180.0
            altitude = 45.0 * np.pi / 180.0
            shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
            hillshade = 255 * (shaded + 1) / 2
            hillshade = hillshade.astype(np.uint8)
            slope_deg = np.degrees(np.arccos(np.sin(slope)))
            aspect_deg = np.degrees(aspect) % 360

        def export_plot(data, title, cmap):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_title(title)
            img = ax.imshow(data, cmap=cmap)
            ax.axis('off')
            plt.colorbar(img, ax=ax, shrink=0.6)
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            st.pyplot(fig)
            st.download_button(f"Download {title} Image", buf, file_name=f"{title}.png", mime="image/png")
            pdf_images.append(buf.getvalue())

        export_plot(hillshade, "Hillshade", "gray")
        export_plot(slope_deg, "Slope (degrees)", "viridis")
        export_plot(aspect_deg, "Aspect (degrees)", "twilight")

        hillshade_path = "/tmp/hillshade.tif"
        with rasterio.open(
            hillshade_path, 'w', driver='GTiff',
            height=hillshade.shape[0], width=hillshade.shape[1],
            count=1, dtype=hillshade.dtype,
            crs=profile['crs'], transform=transform
        ) as dst:
            dst.write(hillshade, 1)

        with open(hillshade_path, "rb") as f:
            st.download_button("Download Hillshade GeoTIFF", f, file_name="hillshade.tif", mime="image/tiff")

        if show_3d:
            lat, lon = 1.25, 103.83
            terrain_layer = pdk.Layer(
                "TerrainLayer",
                elevation_decoder={"rScaler": 1, "gScaler": 1, "bScaler": 1, "offset": 0},
                texture=tmp_tif_path,
                elevation_data=tmp_tif_path,
                bounds=[24550, 27400, 24650, 27500],
                material={"ambient": 0.3, "diffuse": 0.8, "shininess": 32, "specularColor": [255, 255, 255]},
            )
            r = pdk.Deck(layers=[terrain_layer], initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=12, pitch=45))
            st.pydeck_chart(r)

        if st.button("📄 Export PDF Report"):
            pdf = FPDF()
            for img_bytes in pdf_images:
                with NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
                    temp_img.write(img_bytes)
                    temp_img.flush()
                    pdf.add_page()
                    pdf.image(temp_img.name, x=10, y=20, w=190)
            pdf_out = BytesIO()
            pdf.output(pdf_out)
            pdf_out.seek(0)
            st.download_button("Download PDF Report", pdf_out, file_name="GIS_Report.pdf", mime="application/pdf")

    except Exception as e:
        st.warning(f"DEM load or processing error: {e}")

m.to_streamlit(height=700)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Tuple
import httpx
import asyncio
import heapq
import math
import json
 
app = FastAPI(title="Smart Delivery Routing System")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────
 
class Location(BaseModel):
    lat: float
    lng: float
    label: str = ""
 
class RouteRequest(BaseModel):
    stops: List[Location]   # index 0 = depot/start
 
class GeocodeRequest(BaseModel):
    query: str
 
# ─────────────────────────────────────────────
# Haversine distance (km)
# ─────────────────────────────────────────────
 
def haversine(lat1, lng1, lat2, lng2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))
 
# ─────────────────────────────────────────────
# OSRM road-network routing (real roads, free)
# ─────────────────────────────────────────────
 
OSRM_BASE = "https://router.project-osrm.org/route/v1/driving"
 
async def road_route(
    a: Location, b: Location
) -> Tuple[float, List[List[float]]]:
    """
    Returns (distance_km, [[lng,lat], ...] polyline) via OSRM public API.
    Falls back to straight-line if OSRM is unavailable.
    """
    url = (
        f"{OSRM_BASE}/{a.lng},{a.lat};{b.lng},{b.lat}"
        "?overview=full&geometries=geojson"
    )
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
            route = data["routes"][0]
            dist_km = route["distance"] / 1000.0
            coords = route["geometry"]["coordinates"]  # [[lng, lat], ...]
            return dist_km, coords
    except Exception:
        # Fallback: straight-line, two-point polyline
        dist = haversine(a.lat, a.lng, b.lat, b.lng)
        return dist, [[a.lng, a.lat], [b.lng, b.lat]]
 
# ─────────────────────────────────────────────
# Dijkstra's on a complete graph of stops
# ─────────────────────────────────────────────
# We build a weighted complete graph: each node is a stop,
# edge weight = road distance between them.
# Dijkstra finds shortest path from depot to every other node.
# Then we use a nearest-neighbour greedy TSP approximation
# (seeded by Dijkstra distances) for multi-stop ordering.
 
def dijkstra(dist_matrix: List[List[float]], source: int) -> Tuple[List[float], List[int]]:
    """
    Standard Dijkstra on adjacency matrix.
    Returns (distances[], predecessors[]).
    """
    n = len(dist_matrix)
    INF = float("inf")
    dist = [INF] * n
    prev = [-1] * n
    dist[source] = 0.0
    heap = [(0.0, source)]
 
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v in range(n):
            if v == u:
                continue
            w = dist_matrix[u][v]
            if w < INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))
 
    return dist, prev
 
def nearest_neighbour_tsp(dist_matrix: List[List[float]]) -> List[int]:
    """
    Greedy nearest-neighbour TSP starting from node 0 (depot).
    Returns ordered list of node indices forming the tour.
    """
    n = len(dist_matrix)
    if n == 1:
        return [0]
    visited = [False] * n
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = tour[-1]
        best_dist = float("inf")
        best_node = -1
        for j in range(n):
            if not visited[j] and dist_matrix[last][j] < best_dist:
                best_dist = dist_matrix[last][j]
                best_node = j
        tour.append(best_node)
        visited[best_node] = True
    return tour
 
# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────
 
@app.post("/geocode")
async def geocode(req: GeocodeRequest):
    """Forward geocoding via Nominatim."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": req.query,
        "format": "json",
        "limit": 7,
        "addressdetails": 1,
        "accept-language": "en",
        "countrycodes": "in",
    }
    headers = {"User-Agent": "SmartDeliveryRouter/1.0"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            results = r.json()
            return [
                {
                    "display_name": item["display_name"],
                    "lat": float(item["lat"]),
                    "lng": float(item["lon"]),
                }
                for item in results
            ]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Geocoding failed: {e}")
 
 
@app.post("/route")
async def calculate_route(req: RouteRequest):
    """
    1. Fetch real road distances between all stop pairs (OSRM).
    2. Run Dijkstra + nearest-neighbour TSP to find optimal order.
    3. Return ordered stops + full road polylines.
    """
    stops = req.stops
    n = len(stops)
 
    if n < 2:
        raise HTTPException(status_code=400, detail="At least 2 stops required.")
    if n > 12:
        raise HTTPException(status_code=400, detail="Maximum 12 stops supported.")
 
    # ── Build pairwise road distance matrix ──
    # Fetch all unique pairs concurrently
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
 
    async def fetch_pair(i, j):
        dist, coords = await road_route(stops[i], stops[j])
        return i, j, dist, coords
 
    tasks = [fetch_pair(i, j) for i, j in pairs]
    results = await asyncio.gather(*tasks)
 
    INF = float("inf")
    dist_matrix = [[INF] * n for _ in range(n)]
    polyline_cache: dict = {}
 
    for i, j, dist, coords in results:
        dist_matrix[i][j] = dist
        polyline_cache[(i, j)] = coords
 
    # ── Dijkstra from depot (index 0) ──
    dijkstra_dists, dijkstra_prev = dijkstra(dist_matrix, 0)
 
    # ── TSP nearest-neighbour ordering ──
    tour = nearest_neighbour_tsp(dist_matrix)
 
    # ── Build segments for the tour ──
    segments = []
    total_distance = 0.0
    total_duration_min = 0.0
 
    for k in range(len(tour) - 1):
        i, j = tour[k], tour[k + 1]
        d = dist_matrix[i][j]
        total_distance += d
        # Estimate duration: avg city speed 40 km/h
        duration = (d / 40.0) * 60.0
        total_duration_min += duration
        coords = polyline_cache.get((i, j), [[stops[i].lng, stops[i].lat], [stops[j].lng, stops[j].lat]])
        # Convert [lng, lat] → [lat, lng] for Leaflet
        leaflet_coords = [[c[1], c[0]] for c in coords]
        segments.append({
            "from_index": i,
            "to_index": j,
            "from_label": stops[i].label or f"Stop {i+1}",
            "to_label": stops[j].label or f"Stop {j+1}",
            "distance_km": round(d, 2),
            "duration_min": round(duration, 1),
            "polyline": leaflet_coords,
        })
 
    ordered_stops = [
        {
            "index": tour[k],
            "order": k + 1,
            "lat": stops[tour[k]].lat,
            "lng": stops[tour[k]].lng,
            "label": stops[tour[k]].label or f"Stop {tour[k]+1}",
        }
        for k in range(len(tour))
    ]
 
    return {
        "ordered_stops": ordered_stops,
        "segments": segments,
        "total_distance_km": round(total_distance, 2),
        "total_duration_min": round(total_duration_min, 1),
        "dijkstra_distances_from_depot": [round(d, 2) if d != INF else None for d in dijkstra_dists],
        "stop_count": n,
    }
 
 
@app.get("/")
async def serve_index():
    return FileResponse("index.html")
 
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
#include "exporter.h"
#include <fstream>
#include <iostream>
#include <cmath>

void VtkExporter::save_to_vtr(const std::string& filename, const GridInfo& grid, const std::vector<cuComplexType>& sol) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
        return;
    }

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <RectilinearGrid WholeExtent=\"0 " << grid.nx << " 0 " << grid.ny << " 0 " << grid.nz << "\">\n";
    out << "    <Piece Extent=\"0 " << grid.nx << " 0 " << grid.ny << " 0 " << grid.nz << "\">\n";
    
    out << "      <Coordinates>\n";
    
    // X Coordinates
    out << "        <DataArray type=\"Float32\" Name=\"X_COORDINATES\" format=\"ascii\">\n";
    for(double x : grid.x_nodes) out << x << " ";
    out << "\n        </DataArray>\n";

    // Y Coordinates
    out << "        <DataArray type=\"Float32\" Name=\"Y_COORDINATES\" format=\"ascii\">\n";
    for(double y : grid.y_nodes) out << y << " ";
    out << "\n        </DataArray>\n";

    // Z Coordinates
    out << "        <DataArray type=\"Float32\" Name=\"Z_COORDINATES\" format=\"ascii\">\n";
    for(double z : grid.z_nodes) out << z << " ";
    out << "\n        </DataArray>\n";
    
    out << "      </Coordinates>\n";

    out << "      <CellData Scalars=\"E_Magnitude\">\n";
    
    // E_Magnitude
    out << "        <DataArray type=\"Float32\" Name=\"E_Magnitude\" format=\"ascii\">\n";
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                // Average the fields to cell center for visualization
                // This is a simplification. 
                int idx_x = get_dof_lebedev(grid, i, j, k, SubGrid::G000, 0);
                int idx_y = get_dof_lebedev(grid, i, j, k, SubGrid::G000, 1);
                int idx_z = get_dof_lebedev(grid, i, j, k, SubGrid::G000, 2);
                
                float ex = (idx_x < (int)sol.size()) ? sol[idx_x].x : 0.0f;
                float ey = (idx_y < (int)sol.size()) ? sol[idx_y].x : 0.0f;
                float ez = (idx_z < (int)sol.size()) ? sol[idx_z].x : 0.0f;
                
                float mag = std::sqrt(ex*ex + ey*ey + ez*ez);
                out << mag << " ";
            }
        }
    }
    out << "\n        </DataArray>\n";
    out << "      </CellData>\n";
    out << "    </Piece>\n";
    out << "  </RectilinearGrid>\n";
    out << "</VTKFile>\n";
}
// QuPath 0.4.4
import java.io.BufferedReader;
import java.io.FileReader;
import qupath.lib.objects.*
import qupath.lib.roi.*
import qupath.lib.geom.Point2;
def imageData = getCurrentImageData();

// Get location of csv
def file = new File('C://Users//dongw//Documents//TCGA-3M-AB46-01Z-00-DX1.70F638A0-BDCB-4BDE-BBFE-6D78A1A08C5B.csv')

// Create BufferedReader
def csvReader = new BufferedReader(new FileReader(file));
row = csvReader.readLine() // first row (header)

def tmp_points_list = []

// Loop through all the rows of the CSV file.
while ((row = csvReader.readLine()) != null) {
    def rowContent = row.split(",")
    double cx = rowContent[0] as double;
    double cy = rowContent[1] as double;
    tmp_points_list.add(new Point2(cx, cy))
}

// Create annotation
def roi = new PolygonROI(tmp_points_list);
def annotation = new PathAnnotationObject(roi, PathClassFactory.getPathClass("Tumor"));
//imageData.getHierarchy().addPathObject(annotation, true);
imageData.getHierarchy().addPathObject(annotation);
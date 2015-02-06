package tud.ke.ml.project.classifier;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.TreeMap;

import tud.ke.ml.project.framework.classifier.ANearestNeighbor;
import tud.ke.ml.project.util.Pair;

/**
 * This implementation assumes the class attribute is always available (but probably not set)
 * @author cwirth
 *
 */
public class NearestNeighbor extends ANearestNeighbor {
	
	protected double[] scaling;
	protected double[] translation;
	
	private List<List<Object>> trainData;
	
	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		// OK
		Map<Object, Double> votes;
		if (isInverseWeighting()) {
			votes = getWeightedVotes(subset);
		} else {
			votes = getUnweightedVotes(subset);
		}
		return getWinner(votes);
	}
	
	@Override
	protected void learnModel(List<List<Object>> traindata) {
		// Instance based classifier => there is no learning, we just save all instances of the train data 
		this.trainData = traindata;
	}
	
	@Override
	protected Map<Object, Double> getUnweightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		// TODO TEST
		HashMap<Object, Double> map = new HashMap<Object, Double>();
		
		for (Pair<List<Object>, Double> pair : subset) {
			Object instance_class = pair.getA().get(getClassAttribute());
			if (!map.containsKey(instance_class)) {
				map.put(instance_class, new Double(0));
			}
			map.put(instance_class, map.get(instance_class)+1); // TO TEST
		}	
		return map;
	}
	
	@Override
	protected Map<Object, Double> getWeightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		// TODO Auto-generated method stub
		// what is the inverse distance weighting schema ??
		return null;
	}
	
	@Override
	protected Object getWinner(Map<Object, Double> votesFor) {
		// TODO TEST
		// What to do when ex-aequo ?? : When ex-aequo, return the first class found with the highest number of votes
		
		// find the highest Double value
		if ( (votesFor == null) || (votesFor.size() == 0) ) {
			return null;
		}
		Object winner = null;
		for (Entry<Object, Double> e : votesFor.entrySet()) {
			if ( (winner == null) || (e.getValue() > votesFor.get(winner)) ) {
				winner = e.getKey();
			} 
		}
		return winner;
	}
	
	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> testdata) {
		// TO TEST
		int k = getkNearest();
		
		Object data[] = new Object[testdata.size()];
				
		// compute distance from testdata instance to each instance of the trainData
		List<Double> distances = new ArrayList<Double>();
		switch (getMetric()) {
		case 0 :
			// use Manhattan distance
			for (int i=0; i<data.length; i++)  {
				data[i] = new Pair<List<Object>, Double>(trainData.get(i), determineManhattanDistance(testdata, trainData.get(i)));
			}
			break;
		case 1 : 
			// use Euclidian distance
			for (int i=0; i<data.length; i++)  {
				data[i] = new Pair<List<Object>, Double>(trainData.get(i), determineEuclideanDistance(testdata, trainData.get(i)));
			}
			break;
		default :
			throw new RuntimeException("Unhandled metric : "+getMetric());
		}
			
		// pick the getKNearest() better instances : with smallest distances
		Arrays.sort(data, new Comparator<Object>() {
			@Override
			public int compare(Object o1, Object o2) {
				Pair<List<Object>, Double> instance1 = (Pair<List<Object>, Double>) o1;
				Pair<List<Object>, Double> instance2 = (Pair<List<Object>, Double>) o2;
				return instance1.getB().compareTo(instance2.getB());
			}
		});
		
		// get the k first values
		List<Pair<List<Object>, Double>> kNeighbors = new ArrayList<Pair<List<Object>, Double>>();
		for (int i=0; i<k; i++) {
			kNeighbors.add((Pair<List<Object>, Double>)data[i]);
			System.out.println("result["+i+"] = "+((Pair<List<Object>, Double>)data[i]).getB());
		}
		
		// assert kNeighbors.size() == getKNearest() (except if some distances are the same)
		return kNeighbors;
	}
	
	@Override
	protected double determineManhattanDistance(List<Object> instance1,
			List<Object> instance2) {
		int nbAttributes = instance1.size();
		double d = 0;
		for (int i=0; i<nbAttributes; i++) {
			if ( (instance1.get(i) instanceof Double ) && (instance2.get(i) instanceof Double ) ) {
				// TODO first scale the values !!
				d += Math.abs( ((Double)instance1.get(i)) - ((Double)instance2.get(i)) );
			} else if ( (instance1.get(i) instanceof String ) && (instance2.get(i) instanceof Double ) ) {
				d += Math.abs( ((Double)instance1.get(i)) - ((Double)instance2.get(i)) );
			} else {
				// Should not happen
				throw new RuntimeException("Attributes don't pass with each other");
			}
		}
		return d;
	}
	
	@Override
	protected double determineEuclideanDistance(List<Object> instance1,
			List<Object> instance2) {
		// TODO Auto-generated method stub
		return 0;
	}
	
	@Override
	protected double[][] normalizationScaling() {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	protected String[] getMatrikelNumbers() {
		// OK
		String[] numbers = new String[3];
		numbers[0] = "1682731"; // Jan
		numbers[1] = "2805397"; // Killian
		numbers[2] = "2939001"; // Solene
		return numbers;
	}

}
